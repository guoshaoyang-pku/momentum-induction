"""Non-R0 stress test: rules depend on axial count, diagonal count and center.
No group labels are supplied to learned predictors. No future filtering.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cawm.models.directional_metric import LocalMetric, patterns,evidence,query,rollout

def slots():
    b=(np.arange(512)[:,None]>>np.arange(9))&1
    return 25*b[:,4]+5*b[:,[1,3,5,7]].sum(1)+b[:,[0,2,6,8]].sum(1)
SLOTS=slots()

def dataset(seed,n):
    rng=np.random.default_rng(seed)
    rules=rng.integers(0,2,(n,50),dtype=np.uint8)
    x=np.empty((n,16,8,8),dtype=np.uint8)
    x[:,0]=(rng.random((n,8,8))<rng.uniform(.1,.9,(n,1,1)))
    for t in range(15):
        x[:,t+1]=rules[np.arange(n)[:,None,None],SLOTS[patterns(x[:,t])]]
    return x,rules

def prep(x,shift=False):
    c,z=evidence(x[:,:8])
    return tuple(torch.from_numpy(a) for a in (c,z,query(x[:,7],shift),x[:,8].astype(np.float32)))

def logits(table,q):
    return table.gather(1,q.flatten(1)).reshape(q.shape)

@torch.no_grad()
def measure(model,data):
    ce=hit=n=0
    for c,z,q,y in zip(*(v.split(128) for v in data)):
        l=logits(model.table(c,z),q)
        ce+=F.binary_cross_entropy_with_logits(l,y,reduction='sum').item()
        hit+=((l>0)==(y>0)).sum().item(); n+=y.numel()
    return ce/n,hit/n

def score(table,x,shift):
    pred=rollout(table,x[:,:8],shift)
    eq=pred==x[:,8:]
    nonpersistent=(x[:,8:]!=x[:,7:8]).any(axis=(1,2,3))
    return dict(seq8=float(eq.all(axis=(1,2,3)).mean()),
                nonpersistent_n=int(nonpersistent.sum()),
                nonpersistent_seq8=float(eq[nonpersistent].all(axis=(1,2,3)).mean()),
                pixel8=float(eq.mean()))

def oracle(x):
    c,z=evidence(x[:,:8])
    a=(SLOTS[:,None]==SLOTS[None,:]).astype(np.float32)
    m=c@a.T; v=z@a.T
    table=np.where(m>0,v/m.clip(1),0)
    q=query(x[:,7]); t=np.take_along_axis(table,q.reshape(len(q),-1),axis=1).reshape(q.shape)
    known=t!=0
    assert np.array_equal(t[known]>0,x[:,8][known])
    return float((~known).mean()*math.log(2)),float(((t>0)==x[:,8]).mean()),score(table,x,False)

def qa():
    x,r=dataset(101,16)
    # Independent indexed toroidal reference, no reliance on patterns().
    for i in range(16):
        for y in range(8):
            for z in range(8):
                p=x[i,0]; a=sum(p[(y+dy)%8,(z+dx)%8] for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)])
                d=sum(p[(y+dy)%8,(z+dx)%8] for dy in (-1,1) for dx in (-1,1))
                assert x[i,1,y,z]==r[i,25*int(p[y,z])+5*int(a)+int(d)]
    corner=np.zeros((1,8,8),np.uint8);corner[0,7,7]=1
    assert patterns(corner)[0,0,0]==256 and query(corner,True)[0,0,0]==0
    changed=x.copy();changed[:,8:]^=1
    for shift in (False,True):
        a,b=prep(x,shift),prep(changed,shift)
        assert all(torch.equal(u,v) for u,v in zip(a[:3],b[:3])) and not torch.equal(a[3],b[3])
        m=LocalMetric(3);table=m.table(*a[:2]).detach().numpy()
        assert np.array_equal(rollout(table,x[:,:8],shift),rollout(table,changed[:,:8],shift))
        assert np.array_equal(rollout(table,x[:,:8],shift,1),rollout(table,x[:,:8],shift)[:,:1])
    assert len(np.unique(SLOTS))==50
    try:evidence(x)
    except ValueError:pass
    else:raise AssertionError("future accepted")
    oracle(x)
    print("PASS indexed corner/wrap, shifted control, future invariance, exact oracle",flush=True)

def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--M',type=int,default=512)
    ap.add_argument('--valid',type=int,default=2048)
    ap.add_argument('--models',default='exact_group_oracle,frozen_R0_geometry,rank2,rank3,rank3_shift')
    a=ap.parse_args()
    selected=set(a.models.split(','))
    assert selected <= {'exact_group_oracle','frozen_R0_geometry','rank1','rank2','rank3','rank3_shift'} and selected
    assert not a.out.exists()
    torch.set_num_threads(2)
    qa()
    archive=np.load(a.archive)
    for seed in (42,43,44):
        train,tr=dataset(20000+seed,a.M); valid,vr=dataset(30000+seed,a.valid)
        assert not set(map(bytes,tr))&set(map(bytes,vr))
        base=dict(seed=seed,M=a.M,N=64*a.M,valid_n=a.valid,train_sha=digest(train),valid_sha=digest(valid),
                  protocol='unfiltered directional count CA, synthetic distribution, non-R0',device='cpu')
        def emit(r):
            with a.out.open('a') as f:f.write(json.dumps(base|r,allow_nan=False)+'\n')
            print(seed,r['model'],r['valid_bce'],r['rollout']['nonpersistent_seq8'],flush=True)
        if 'exact_group_oracle' in selected:
            tl,tp,_=oracle(train);vl,vp,ro=oracle(valid)
            emit(dict(model='exact_group_oracle',train_bce=tl,train_pixel=tp,valid_bce=vl,valid_pixel=vp,rollout=ro,params=0,steps=0))
        for name,rank,shift in [('frozen_R0_geometry',2,False),('rank1',1,False),('rank2',2,False),('rank3',3,False),('rank3_shift',3,True)]:
            if name not in selected: continue
            torch.manual_seed(seed)
            frozen=archive[f'metric_M2048_s{seed}_shift0'] if name.startswith('frozen') else None
            model=LocalMetric(rank,frozen);opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.01)
            t=prep(train,shift);v=prep(valid,shift)
            rng=torch.Generator().manual_seed(seed+100)
            trace=[]
            for step in range(1,a.steps+1):
                ix=torch.randperm(a.M,generator=rng)[:32]
                c,z,q,y=(u[ix] for u in t)
                loss=F.binary_cross_entropy_with_logits(logits(model.table(c,z),q),y)
                opt.zero_grad();loss.backward();opt.step()
                if step in (1,a.steps//2,a.steps):trace.append(dict(step=step,train_bce=measure(model,t)[0]))
            tl,tp=measure(model,t);vl,vp=measure(model,v)
            with torch.no_grad():table=model.table(*v[:2]).numpy()
            emit(dict(model=name,rank=rank,shift=shift,train_bce=tl,train_pixel=tp,valid_bce=vl,valid_pixel=vp,rollout=score(table,valid,shift),params=sum(p.numel() for p in model.parameters() if p.requires_grad),allocated_params=sum(p.numel() for p in model.parameters()),steps=a.steps,trace=trace,weight=model.encoder.weight.detach().tolist(),gain=float(F.softplus(model.gain).detach()),bias=float(model.bias.detach()),archive_sha=hashlib.sha256(a.archive.read_bytes()).hexdigest(),certified_converged=False))

if __name__=='__main__':main()
