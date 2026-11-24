#!/usr/bin/env python3
"""Train a support-shrunk confidence model, with L3-SC and unfiltered controls.
Evaluation coverage is diagnostic only; no future information enters inference.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.data import sample_batch
from cawm.rules import rule_split
from run_raw_evidence_experiment import original,prep,measure,digest
from run_raw_evidence_support import CORPORA,diagnose

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpora",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corp={}
    for k,(name,sha) in CORPORA.items():
        path=a.corpora/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        corp[k]=np.load(path)["frames"]
    with a.out.open("x") as f:
        for seed in (42,43,44):
            train_sc=original(seed,512)
            _,train_unfiltered,_=sample_batch(range(512),1000+seed,rule_split()[0],grid=8)
            for dist,train,modes in (("L3-SC",train_sc,("support",)),("L3-unfiltered",train_unfiltered,("mean","count","support"))):
                for mode in modes:
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
                    row=dict(seed=seed,mode=mode,distribution=dist,M=512,N=32768,updates=2400,params=model.trainable_param_count(),train_sha=digest(train),train_bce=bce,train_pixel=pixel,trace=trace,certified=False,eval={k:diagnose(model,v) for k,v in corp.items()},elapsed_s=time.monotonic()-start,prior_count=float(F.softplus(model.log_prior_count).detach()) if mode=="support" else None)
                    f.write(json.dumps(row,allow_nan=False,sort_keys=True)+"\n");f.flush()
                    print(dist,mode,seed,bce,{k:(v["bce"],v["seq8"],v["unseen_slot"]["bce"]) for k,v in row["eval"].items()},flush=True)
if __name__=="__main__":main()
