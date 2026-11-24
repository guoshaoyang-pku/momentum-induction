"""Learn raw neighborhood affinity with auxiliary masked-history BCE.
Freeze groups from learned weights, then decode persistent latent rule bits.
Full held-out released corpora, no future-dependent selection for L3-unfiltered.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.models.rule_hypotheses import decode,learned_classes,outer_totalistic_classes,prefix_table,raw_pattern
from run_raw_evidence_experiment import original,prep,measure,digest
from run_raw_evidence_support import CORPORA,diagnose

def fit(seed,steps):
    x=original(seed,512)
    torch.manual_seed(seed);m=RawEvidenceCA(mode="mean")
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    c,z,q,y=prep(m,x)
    h=torch.from_numpy(x)
    # Each held-out historical transition is reconstructed with evidence from
    # the other six. This is an auxiliary training objective, not causal
    # online prediction of that historical frame.
    pat=m.pattern(h[:,:7]).flatten(2)
    obs=h[:,1:8].flatten(2).float()
    cs=[];zs=[]
    for t in range(7):
        cc=torch.zeros_like(c);zz=torch.zeros_like(z)
        cc.scatter_add_(1,pat[:,t],torch.ones_like(obs[:,t]))
        zz.scatter_add_(1,pat[:,t],2*obs[:,t]-1)
        cs.append(c-cc);zs.append(z-zz)
    rng=torch.Generator().manual_seed(seed+100)
    trace=[];step=0
    for _ in range(steps//16):
        for ix in torch.randperm(512,generator=rng).split(32):
            t=int(torch.randint(7,(1,),generator=rng))
            w=m.kernel()
            # reuse the same learned affinity for both objectives
            def logits(cc,zz,query):
                table=F.softplus(m.log_gain)*(zz@w.T)/(cc@w.T).clamp_min(1e-8)+m.bias
                return m.gather(table,query)
            loss=F.binary_cross_entropy_with_logits(logits(c[ix],z[ix],q[ix]),y[ix])
            aux=F.binary_cross_entropy_with_logits(logits(cs[t][ix],zs[t][ix],pat[ix,t]),obs[ix,t])
            opt.zero_grad();(loss+.5*aux).backward();opt.step();step+=1
            if step%400==0 or step==steps:trace.append(dict(step=step,train_bce=measure(m,(c,z,q,y))[0]))
    return m,dict(seed=seed,updates=step,M=512,N=32768,params=m.trainable_param_count(),train_sha=digest(x),train_bce=measure(m,(c,z,q,y))[0],trace=trace,certified=False)

def evaluate(x,classes,steps=8,budget=512,topk=4):
    k1=0;k4=0;zero=0;conflicts=0;exhausted=0;no_path=0
    possible=0;mass=0.;nodes=0;missing=0;paths_n=0
    frame_hit=np.zeros(steps,np.int64);zero_hit=np.zeros(steps,np.int64)
    unknown_pixel=0;first_count=0;first_gain=0.;wrong_known=0
    for seq in x:
        prefix=seq[:8];target=seq[8:8+steps]
        t=prefix_table(prefix,classes)
        if t is None: conflicts+=1;continue
        # zero-fill reference uses same partition and persistent rule.
        q=classes[raw_pattern(prefix[-1])]
        p=t[q];unknown_pixel+=int((p<0).sum());first_count+=p.size
        wrong_known+=int(((p>=0)&(p!=target[0])).sum())
        first_gain+=int((p>=0).sum())
        tt=np.maximum(t,0);cur=prefix[-1];z=[]
        for _ in range(steps):
            cur=tt[classes[raw_pattern(cur)]].astype(np.uint8);z.append(cur)
        zgood=(np.asarray(z)==target).reshape(steps,-1).all(1).cumprod()
        zero_hit+=zgood;zero+=int(zgood[-1])
        # Whether ANY persistent assignment can produce observed future.
        # Post-hoc only, never used to select decoded paths.
        whole=classes[raw_pattern(seq[:7+steps])].ravel()
        yy=seq[1:8+steps].ravel()
        tmp=np.full_like(t,-1);tmp[whole]=yy
        possible+=int(np.array_equal(tmp[whole],yy))
        r=decode(prefix,classes,steps,topk,budget)
        nodes+=r["nodes"];exhausted+=int(r["exhausted"]);paths_n+=len(r["paths"])
        if not r["paths"]:no_path+=1;continue
        best=r["paths"][0]
        good=(best["frames"]==target).reshape(steps,-1).all(1).cumprod()
        frame_hit+=good;k1+=int(good[-1])
        k4+=int(any(np.array_equal(a["frames"],target) for a in r["paths"]))
        mass+=sum(a["probability"] for a in r["paths"])
        missing+=best["new_bits"]
    n=len(x)
    return dict(n=n,sha=digest(x),steps=steps,groups=int(classes.max())+1,
                zero_seq=zero/n,map_seq=k1/n,top4_seq=k4/n,conflicts=conflicts,
                exhausted=exhausted,no_path=no_path,possible=possible/n,
                mean_top4_mass=mass/n,mean_nodes=nodes/n,mean_paths=paths_n/n,
                mean_map_new_bits=missing/max(1,n-conflicts-no_path),
                cumulative_map=(frame_hit/n).tolist(),cumulative_zero=(zero_hit/n).tolist(),
                unknown_first_pixels=unknown_pixel,scored_first_pixels=first_count,
                wrong_known_first=wrong_known,known_first_bits=first_gain)

def main():
    p=argparse.ArgumentParser();p.add_argument("--corpora",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=1200);a=p.parse_args()
    assert a.steps>0 and a.steps%16==0
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corp={}
    for k,(name,sha) in CORPORA.items():
        path=a.corpora/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        corp[k]=np.load(path)["frames"]
    ref=outer_totalistic_classes()
    with a.out.open("x") as f:
        def emit(r):
            f.write(json.dumps(r,allow_nan=False,sort_keys=True)+"\n");f.flush()
            print(r["arm"],r.get("seed"),{k:(round(v["map_seq"],4),round(v["top4_seq"],4),v["conflicts"],v["exhausted"]) for k,v in r["eval"].items()},flush=True)
        emit(dict(arm="known_partition_uniform_prior",eval={k:evaluate(x,ref) for k,x in corp.items()}))
        for seed in (42,43,44):
            m,r=fit(seed,a.steps)
            groups=learned_classes(m.kernel().detach().numpy(),.5)
            eq=groups[:,None]==groups[None,:];true=ref[:,None]==ref[None,:]
            r.update(arm="learned_partition_masked_history",class_map=groups.tolist(),
                     pair_false_merge=int((eq&~true).sum()),pair_false_split=int((~eq&true).sum()),
                     greedy={k:diagnose(m,x) for k,x in corp.items()},
                     eval={k:evaluate(x,groups) for k,x in corp.items()})
            emit(r)
if __name__=="__main__":main()
