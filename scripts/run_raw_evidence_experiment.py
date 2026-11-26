#!/usr/bin/env python3
"""Finite-budget, independent fresh fits on released L3-SC and unfiltered noise.
No raw corpus/checkpoints written. Fixed prefix-only rule inference.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.data import ChunkedStreamDataset, _draw_sample_params
from cawm.rules import rule_split
from cawm.simulate import ca_step, rule_tables

def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()[:20]

def noisy(seed,p,half,n):
    pool=rule_split()[0 if half=="train" else 1]
    master=1000+seed+(0 if half=="train" else 10000)
    a=[_draw_sample_params(i,master,pool,8) for i in range(n)]
    rule,init=zip(*a)
    rng=np.random.default_rng([20261124,seed+(0 if half=="train" else 9000),int(round(p*1e6))])
    tab=rule_tables(rule)
    x=np.empty((n,16,8,8),dtype=np.uint8);x[:,0]=np.stack(init)
    for t in range(15):
        x[:,t+1]=ca_step(x[:,t],tab)^(rng.random((n,8,8))<p).astype(np.uint8)
    return x

def original(seed,max_m):
    ds=ChunkedStreamDataset(master_seed=1000+seed,half="train",grid=8,n_chunks=(max_m+63)//64,chunk_size=64,sc_filter=True,sc_oversample=4)
    return np.concatenate([ds[i]["frames"].numpy() for i in range(len(ds))])[:max_m]

def prep(m,x):
    x=torch.from_numpy(x)
    return (*m.evidence(x[:,:8]),m.query(x[:,7]),x[:,8].float())

@torch.no_grad()
def measure(m,data):
    c,z,q,y=data
    logits=m.gather(m.table(c,z),q)
    return float(F.binary_cross_entropy_with_logits(logits,y)),float(((logits>0)==(y>0)).float().mean())

@torch.no_grad()
def roll_metrics(m,x):
    hits=[]
    for batch in np.array_split(x,max(1,math.ceil(len(x)/128))):
        pred=m.rollout((torch.from_numpy(batch[:,:8]).float()*2-1)[:,None]).numpy()
        hits.append(pred==batch[:,8:])
    hit=np.concatenate(hits)
    dynamic=(x[:,8:]!=x[:,7:8]).any(axis=(1,2,3))
    exact=hit.reshape(len(x),8,-1).all(2).cumprod(1)
    return dict(n=len(x),dynamic_n=int(dynamic.sum()),pixel_by_frame=hit.mean(axis=(0,2,3)).tolist(),
                cumulative_exact=exact.mean(0).tolist(),dynamic_seq8=float(exact[dynamic,-1].mean()) if dynamic.any() else None)

def run(train,valid,l1,seed,M,mode,shift,steps):
    torch.manual_seed(seed)
    model=RawEvidenceCA(mode=mode,shifted=shift)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    tr=prep(model,train);va=prep(model,valid[:256])
    rng=torch.Generator().manual_seed(seed+100)
    nsteps=math.ceil(steps/math.ceil(M/32))*math.ceil(M/32)
    step=0;trace=[];start=time.monotonic()
    while step<nsteps:
        for ix in torch.randperm(M,generator=rng).split(32):
            c,z,q,y=(a[ix] for a in tr)
            loss=F.binary_cross_entropy_with_logits(model.gather(model.table(c,z),q),y)
            opt.zero_grad();loss.backward();opt.step()
            step+=1
            if step%200==0 or step==nsteps:
                trace.append(dict(step=step,train_bce=measure(model,tr)[0],val_bce=measure(model,va)[0]))
    tl,tp=measure(model,tr);vl,vp=measure(model,va)
    return dict(M=M,N=64*M,seed=seed,mode=mode,shift=shift,params=model.trainable_param_count(),updates=step,
        train_bce=tl,train_pixel=tp,val_bce=vl,val_pixel=vp,C_bits=64*M*(1-tl/math.log(2)),
        train_sha=digest(train),eval_sha=digest(valid[:256]),rollout_sha=digest(valid),trace=trace,learned_eps=float(.5*model.noise_logit.detach().sigmoid()) if mode=="count" else None,
        rollout=roll_metrics(model,valid),l1_rollout=roll_metrics(model,l1),elapsed_s=time.monotonic()-start,certified=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--l3",type=Path,required=True);p.add_argument("--l1",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=600)
    p.add_argument("--sizes",default="128,512,2048");p.add_argument("--seeds",default="42,43,44")
    p.add_argument("--noise",default=".1,.5");p.add_argument("--mode",default="mean,count")
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sizes=sorted(set(map(int,a.sizes.split(","))));seeds=list(map(int,a.seeds.split(",")))
    modes=a.mode.split(",")
    assert hashlib.sha256(a.l3.read_bytes()).hexdigest()=="d65320dc0db22130c5e2b69ebaf6da617003a3f8545be52f4325335392181801"
    assert hashlib.sha256(a.l1.read_bytes()).hexdigest()=="d7ea43b829ef07fa2b5b628e88f2979df3de74cfe800fb04ca4211a88f8ad5a1"
    l3=np.load(a.l3)["frames"];l1=np.load(a.l1)["frames"]
    # Whole released L3 / L1 corpora in final rollouts, first 256 for validation loss.
    assert not a.out.exists()
    with a.out.open("x") as f:
        for p in [None]+list(map(float,a.noise.split(","))):
            for seed in seeds:
                tr=original(seed,max(sizes)) if p is None else noisy(seed,p,"train",2048)
                va=l3 if p is None else noisy(seed,p,"zs",256)
                for M in sizes if p is None else [512]:
                    for mode,shift in [(m,False) for m in modes]+([("count",True)] if p is None and M==512 and "count" in modes else []):
                        r=run(tr[:M],va,l1,seed,M,mode,shift,a.steps)
                        r.update(p=p,distribution="L3-SC" if p is None else "unfiltered-noise")
                        f.write(json.dumps(r,allow_nan=False,sort_keys=True)+"\n");f.flush()
                        print(json.dumps({k:r[k] for k in ("distribution","p","M","seed","mode","shift","train_bce","val_bce","learned_eps","elapsed_s")}|{"seq8":r["rollout"]["cumulative_exact"][-1],"dynamic_seq8":r["rollout"]["dynamic_seq8"]}),flush=True)
if __name__=="__main__":main()
