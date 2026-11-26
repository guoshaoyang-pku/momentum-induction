"""Reconstruct full train/validation BCE and autonomous predictions from stored parameters."""
import json,math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
from run_directional_metric import dataset,digest
from cawm.models.directional_metric import patterns
ROOT=Path(__file__).resolve().parents[1]/'results/directional_metric'
rows=[json.loads(l) for f in ROOT.glob('*.jsonl') for l in f.read_text().splitlines()]
assert len(rows)==42 and len({(r['M'],r['seed'],r['model']) for r in rows})==42
raw=((np.arange(512)[:,None]>>np.arange(9))&1)*2-1
b=(raw+1)//2
groups=25*b[:,4]+5*b[:,[1,3,5,7]].sum(1)+b[:,[0,2,6,8]].sum(1)
maxerr=0
maxlogit=0
rounding_bits=0
for r in rows:
    for split,n,seedbase in [('train',r['M'],20000),('valid',r['valid_n'],30000)]:
        x,_=dataset(seedbase+r['seed'],n)
        assert digest(x)==r[split+'_sha']
        p=patterns(x[:,:7]).reshape(n,-1)
        y=x[:,1:8].reshape(n,-1).astype(np.float64)*2-1
        counts=np.zeros((n,512));votes=counts.copy()
        np.add.at(counts,(np.arange(n)[:,None],p),1)
        np.add.at(votes,(np.arange(n)[:,None],p),y)
        exact=r['model']=='exact_group_oracle'
        if exact:
            k=(groups[:,None]==groups[None,:]).astype(float)
            mass=counts@k.T
            table=np.divide(votes@k.T,mass,out=np.zeros_like(mass),where=mass>0)
        else:
            emb=raw@np.asarray(r['weight']).T
            k=np.exp(-((emb[:,None]-emb[None,:])**2).sum(-1))
            table=r['gain']*(votes@k.T)/(counts@k.T).clip(1e-6)+r['bias']
        def predict(cur,tab=table):
            q=patterns(cur)
            if r.get('shift',False):q=np.roll(q,(4,4),axis=(-2,-1))
            return tab[np.arange(n)[:,None,None],q]
        l=predict(x[:,7])
        if exact:
            known=l!=0
            assert np.array_equal(l[known]>0,x[:,8][known])
            ce=(~known).mean()*math.log(2)
        else:
            ce=np.logaddexp(0,l*(1-2*x[:,8].astype(float))).mean()
        err=abs(ce-r[split+'_bce']);maxerr=max(maxerr,err)
        assert err<2e-6,(r['model'],r['seed'],r['M'],split,err)
        if exact:
            tab32=table
            l32=l
        else:
            # Restore the exact training/evaluation float32 operation order.
            a=torch.nn.functional.linear(torch.from_numpy(raw.astype(np.float32)),torch.tensor(r['weight'],dtype=torch.float32))
            kk=(-(a[:,None,:]-a[None,:,:]).square().sum(-1)).exp()
            c=torch.from_numpy(counts.astype(np.float32));z=torch.from_numpy(votes.astype(np.float32))
            def calc(c,z):return r['gain']*(z@kk.T)/(c@kk.T).clamp_min(1e-6)+r['bias']
            # The reported BCE/pixel were evaluated with 128-world batches.
            t32=torch.cat([calc(c0,z0) for c0,z0 in zip(c.split(128),z.split(128))]).numpy()
            l32=predict(x[:,7],t32)
            tab32=calc(c,z).numpy()
            maxlogit=max(maxlogit,float(np.max(abs(l-l32))))
            changed=(l>0)!=(l32>0)
            rounding_bits+=int(changed.sum())
            assert np.all(abs(l[changed])<1e-4)
        assert abs(((l32>0)==x[:,8]).mean()-r[split+'_pixel'])<1e-9,(r['model'],r['seed'],r['M'],split)
        if split=='valid':
            cur=x[:,7];eq=[]
            for _ in range(8):
                cur=(predict(cur,tab32)>0).astype(np.uint8)
                eq.append(cur==x[:,8+len(eq)])
            eq=np.stack(eq,axis=1)
            active=(x[:,8:]!=x[:,7:8]).any(axis=(1,2,3))
            assert abs(eq.all(axis=(1,2,3)).mean()-r['rollout']['seq8'])<1e-9
            assert abs(eq[active].all(axis=(1,2,3)).mean()-r['rollout']['nonpersistent_seq8'])<1e-9
print('PASS 42 records, 84 full-data BCE reconstructions, 42 closed-loop reconstructions; maximum BCE discrepancy',maxerr, 'max first-frame logit discrepancy',maxlogit,'near-zero sign changes',rounding_bits)
