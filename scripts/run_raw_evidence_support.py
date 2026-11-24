#!/usr/bin/env python3
"""Compare support normalization on L3-SC and measure transfer by coverage.
Train exclusively on original L3-SC with fixed M=512, 2400 updates.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.simulate import neighbor_count
from run_raw_evidence_experiment import original,prep,measure,digest

CORPORA={
    "L3-SC":("l3_zs_sc_g8.npz","d65320dc0db22130c5e2b69ebaf6da617003a3f8545be52f4325335392181801"),
    "L3-unfiltered":("l3_zs_g8.npz","d534844cab4fa5f15447e26142bf32c4d861bfcd6542c80f45170c0e97eda4fc"),
    "L1":("l1_val_gol_g8.npz","d7ea43b829ef07fa2b5b628e88f2979df3de74cfe800fb04ca4211a88f8ad5a1"),
}

def coverage(x):
    b,t,h,w=x[:,:15].shape
    s=9*x[:,:15].astype(np.int16)+neighbor_count(x[:,:15].reshape(b*t,h,w)).reshape(b,t,h,w)
    observed=np.zeros((b,18),bool)
    np.put_along_axis(observed,s[:,:7].reshape(b,-1),True,axis=1)
    seen=observed[np.arange(b)[:,None,None,None],s[:,7:]]
    raw=RawEvidenceCA.pattern(torch.from_numpy(x[:,:15])).numpy()
    observed_raw=np.zeros((b,512),bool)
    np.put_along_axis(observed_raw,raw[:,:7].reshape(b,-1),True,axis=1)
    seen_raw=observed_raw[np.arange(b)[:,None,None,None],raw[:,7:]]
    return seen,seen_raw

@torch.no_grad()
def diagnose(model,x):
    seen,seen_raw=coverage(x)
    full=seen.reshape(len(x),-1).all(1)
    first=seen[:,0];rawfirst=seen_raw[:,0];y=x[:,8].astype(bool)
    logits=[];roll=[]
    for b in np.array_split(x,max(1,math.ceil(len(x)/128))):
        p=(torch.from_numpy(b[:,:8]).float()*2-1)[:,None]
        logits.append(model.first(p).numpy());roll.append(model.rollout(p).numpy())
    logit=np.concatenate(logits);pred=np.concatenate(roll)
    good=(logit>0)==y;ce=np.logaddexp(0,logit)-y*logit
    hit=pred==x[:,8:]
    exact=hit.reshape(len(x),8,-1).all(2).cumprod(1)
    dynamic=(x[:,8:]!=x[:,7:8]).reshape(len(x),-1).any(1)
    def slice_stats(mask):
        return dict(n=int(mask.sum()),accuracy=float(good[mask].mean()) if mask.any() else None,
                    bce=float(ce[mask].mean()) if mask.any() else None)
    def seq(mask):
        return dict(n=int(mask.sum()),seq8=float(exact[mask,-1].mean()) if mask.any() else None)
    return dict(n=len(x),sha=digest(x),bce=float(ce.mean()),pixel=float(good.mean()),seq8=float(exact[:,-1].mean()),
                seen_slot=slice_stats(first),unseen_slot=slice_stats(~first),
                seen_slot_unseen_raw=slice_stats(first&~rawfirst),seen_raw=slice_stats(rawfirst),
                fully_covered=seq(full),incomplete=seq(~full),dynamic=seq(dynamic),
                cumulative_exact=exact.mean(0).tolist(),pixel_by_frame=hit.mean(axis=(0,2,3)).tolist())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpora",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corp={}
    for k,(f,sha) in CORPORA.items():
        path=a.corpora/f
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        corp[k]=np.load(path)["frames"]
    assert coverage(corp["L3-SC"])[0].all()
    with a.out.open("x") as f:
        for seed in (42,43,44):
            train=original(seed,512)
            for mode in ("mean","count","sqrt"):
                start=time.monotonic()
                torch.manual_seed(seed);model=RawEvidenceCA(mode=mode)
                opt=torch.optim.Adam(model.parameters(),lr=.01)
                tr=prep(model,train);rng=torch.Generator().manual_seed(seed+100)
                trace=[]
                for epoch in range(150):
                    for ix in torch.randperm(512,generator=rng).split(32):
                        c,z,q,y=(v[ix] for v in tr)
                        loss=F.binary_cross_entropy_with_logits(model.gather(model.table(c,z),q),y)
                        opt.zero_grad();loss.backward();opt.step()
                    if (epoch+1)%25==0:
                        trace.append(dict(updates=(epoch+1)*16,train_bce=measure(model,tr)[0]))
                bce,pixel=measure(model,tr)
                row=dict(seed=seed,mode=mode,M=512,N=32768,updates=2400,params=model.trainable_param_count(),train_sha=digest(train),train_bce=bce,train_pixel=pixel,trace=trace,certified=False,eval={k:diagnose(model,v) for k,v in corp.items()},elapsed_s=time.monotonic()-start)
                f.write(json.dumps(row,allow_nan=False,sort_keys=True)+"\n");f.flush()
                print(mode,seed,bce,{k:(v["bce"],v["seq8"],v["fully_covered"],v["incomplete"]) for k,v in row["eval"].items()},flush=True)
if __name__=="__main__":main()
