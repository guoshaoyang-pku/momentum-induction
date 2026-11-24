"""Train grid-8 vs mixed grid models; evaluate untouched released SC corpora
and separately generated 32-step autonomous futures. No tuning on validation.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.data import ChunkedStreamDataset,_draw_sample_params
from cawm.rules import rule_split
from cawm.simulate import simulate,neighbor_count
from run_raw_evidence_experiment import prep,digest
from run_raw_evidence_support import diagnose

SHAS={4:"6efe84e4c4fddb0be5c8309ec727de525bf3ec69db305639fa626d96d72cf183",8:"d65320dc0db22130c5e2b69ebaf6da617003a3f8545be52f4325335392181801",16:"14a1857770382869c06e40b0808377a4dd77b3daa6f73c6d28e46294ccb38331"}

def train_data(seed,g,n):
    ds=ChunkedStreamDataset(master_seed=1000+seed,half="train",grid=g,n_chunks=n//64,chunk_size=64,sc_filter=True,sc_oversample=4)
    return np.concatenate([ds[i]["frames"].numpy() for i in range(len(ds))])

def long_data(g):
    samples=[_draw_sample_params(i,2601125,rule_split()[1],g) for i in range(512)]
    rules,init=zip(*samples)
    return simulate(rules,np.stack(init),39)

@torch.no_grad()
def long_eval(m,x):
    good=[];covered=[]
    for batch in np.array_split(x,8):
        p=(torch.from_numpy(batch[:,:8]).float()*2-1)[:,None]
        pred=m.rollout(p,steps=32).numpy()
        good.append(pred==batch[:,8:])
        src=batch[:,:39];b,t,h,w=src.shape
        s=9*src.astype(np.int16)+neighbor_count(src.reshape(b*t,h,w)).reshape(b,t,h,w)
        obs=np.zeros((b,18),bool)
        np.put_along_axis(obs,s[:,:7].reshape(b,-1),True,axis=1)
        covered.append(obs[np.arange(b)[:,None,None,None],s[:,7:]].reshape(b,-1).all(1))
    hit=np.concatenate(good);full=np.concatenate(covered)
    ex=hit.reshape(len(x),32,-1).all(2).cumprod(1)
    return dict(n=len(x),sha=digest(x),covered_n=int(full.sum()),covered_seq32=float(ex[full,-1].mean()) if full.any() else None,cumulative_exact=ex.mean(0).tolist(),pixel=hit.mean(axis=(0,2,3)).tolist())

def train(seed,arm,steps):
    torch.manual_seed(seed);m=RawEvidenceCA(mode="mean")
    counts={8:512} if arm=="grid8" else {4:128,8:256,16:128}
    raw={g:train_data(seed,g,n) for g,n in counts.items()}
    data={g:prep(m,x) for g,x in raw.items()}
    rng=torch.Generator().manual_seed(seed+100);opt=torch.optim.Adam(m.parameters(),lr=.01)
    trace=[];step=0
    def full():
        with torch.no_grad():
            out={}
            for g,(c,z,q,y) in data.items():
                out[str(g)]=float(bce(m.gather(m.table(c,z),q),y))
            return out
    # Every epoch contains exactly 16 batches of 32 trajectories.
    # Grid-balanced arm samples each trajectory once; objective mean cell BCE
    # within each batch, hence equally weighted trajectories across grids.
    while step<steps:
        batches=[(g,ix) for g,n in counts.items() for ix in torch.randperm(n,generator=rng).split(32)]
        for i in torch.randperm(len(batches),generator=rng):
            g,ix=batches[int(i)]
            c,z,q,y=(v[ix] for v in data[g])
            loss=bce(m.gather(m.table(c,z),q),y)
            opt.zero_grad();loss.backward();opt.step();step+=1
            if step%400==0:trace.append(dict(step=step,train_bce_by_grid=full()))
    return m,dict(seed=seed,arm=arm,updates=step,counts=counts,params=m.trainable_param_count(),train_sha={g:digest(x) for g,x in raw.items()},train_bce_by_grid=full(),trace=trace,certified=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--corpora",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=1200);a=p.parse_args()
    assert a.steps>0 and a.steps%16==0
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corp={}
    for g,sha in SHAS.items():
        path=a.corpora/f"l3_zs_sc_g{g}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        corp[g]=np.load(path)["frames"]
    long={g:long_data(g) for g in SHAS}
    with a.out.open("x") as f:
        for seed in (42,43,44):
            for arm in ("grid8","mixed"):
                start=time.monotonic();m,r=train(seed,arm,a.steps)
                r.update(eval={g:diagnose(m,x) for g,x in corp.items()},long={g:long_eval(m,x) for g,x in long.items()},elapsed_s=time.monotonic()-start)
                f.write(json.dumps(r,allow_nan=False,sort_keys=True)+"\n");f.flush()
                print(seed,arm,{g:(r["eval"][g]["seq8"],r["long"][g]["cumulative_exact"][-1]) for g in corp},flush=True)
if __name__=="__main__":main()
