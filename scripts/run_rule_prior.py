"""Train-only empirical rule priors and independent-bit predictive diagnostics."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from cawm.models.rule_hypotheses import decode,prefix_table,raw_pattern,merge_observed_classes
from run_raw_evidence_experiment import original,noisy,digest
from run_raw_evidence_support import CORPORA

def estimate(prefixes,classes):
    """One vote per observed rule bit per world, not per pixel occurrence."""
    if np.asarray(prefixes).ndim!=4 or prefixes.shape[1]!=8:
        raise ValueError("8-frame histories required")
    tables=[prefix_table(x,classes) for x in prefixes]
    if not tables or any(t is None for t in tables):raise ValueError("empty histories or contradictory partition")
    t=np.stack(tables)
    n=(t>=0).sum(0);k=(t==1).sum(0)
    return (k+1)/(n+2),n

def score_first(prefix,target,classes,prior):
    """Exact conditional joint likelihood for a frame under the persistent-bit model.
    Unknown repeated queries share ONE bit. Per-pixel marginal NLL is separate.
    Return None for impossible joint likelihood; never clamp it into finite NLL.
    """
    t=prefix_table(prefix,classes)
    if t is None:return None,None
    q=classes[raw_pattern(prefix[-1])]
    y=np.asarray(target)
    if y.shape!=q.shape or not np.isin(y,[0,1]).all():raise ValueError("binary target")
    p=np.where(t[q]<0,prior[q],t[q])
    likelihood=np.where(y==1,p,1-p)
    marginal=None if (likelihood<=0).any() else float(-np.log(likelihood).sum())
    loss=0.
    for j in np.unique(q):
        yy=y[q==j]
        if (yy!=yy[0]).any():return None,marginal
        if t[j]>=0:
            if t[j]!=yy[0]:return None,marginal
        else:loss-=math.log(prior[j] if yy[0] else 1-prior[j])
    return loss,marginal

def evaluate(x,c,prior):
    joint=marginal=0.;bad_joint=bad_marginal=0;map_n=top4=exhausted=0;mass=0.
    for seq in x:
        a,b=score_first(seq[:8],seq[8],c,prior)
        if a is None:bad_joint+=1
        else:joint+=a
        if b is None:bad_marginal+=1
        else:marginal+=b
        r=decode(seq[:8],c,8,4,512,prior=prior)
        exhausted+=r["exhausted"]
        if r["paths"]:
            hits=[np.array_equal(z["frames"],seq[8:]) for z in r["paths"]]
            map_n+=hits[0];top4+=any(hits);mass+=sum(z["probability"] for z in r["paths"])
    n=len(x)
    return dict(n=n,sha=digest(x),joint_nll_per_target=None if bad_joint else joint/(64*n),
        marginal_nll_per_target=None if bad_marginal else marginal/(64*n),
        impossible_joint=bad_joint,impossible_marginal=bad_marginal,map_seq=map_n/n,top4_seq=top4/n,
        exhausted=exhausted,mean_top4_mass=mass/n)

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--corpora",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corp={}
    for key,(name,sha) in CORPORA.items():
        path=a.corpora/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        corp[key]=np.load(path)["frames"]
    rows=[json.loads(l) for l in a.input.read_text().splitlines()]
    canonical=None
    with a.out.open("x") as f:
        for r in rows:
            if "class_map" not in r:continue
            seed=r["seed"];train=original(seed,512);assert digest(train)==r["train_sha"]
            c,merges=merge_observed_classes(np.array(r["class_map"]),train[:,:8],min_overlap=12)
            # canonical class IDs by minimum raw pattern; independent of merge order.
            _,ind=np.unique(c,return_index=True);order=np.argsort(ind);remap=np.argsort(order);c=remap[c]
            if canonical is None:canonical=c.copy()
            elif not np.array_equal(c,canonical):raise ValueError("partitions differ; cannot reuse uniform baseline")
            arms=[("uniform",np.full(int(c.max())+1,.5),None,None)]
            for name,xx in [("SC-history",train),("unfiltered-history",noisy(seed,0.,"train",512))]:
                q,n=estimate(xx[:,:8],c);arms.append((name,q,n,digest(xx[:,:8])))
            for name,q,n,sha in arms:
                # all three recovered partitions have same canonical identity;
                # uniform already evaluated on seed42. No pseudo-replication.
                if seed!=42 and name=="uniform":continue
                out=dict(seed=seed,arm=name,classes_sha=digest(c.astype(np.int16)),groups=int(c.max())+1,
                         prior=q.tolist(),observations=None if n is None else n.tolist(),prior_history_sha=sha,
                         eval={k:evaluate(x,c,q) for k,x in corp.items()})
                f.write(json.dumps(out,allow_nan=False,sort_keys=True)+"\n");f.flush()
                print(seed,name,{k:(v["map_seq"],v["top4_seq"],v["joint_nll_per_target"],v["marginal_nll_per_target"]) for k,v in out["eval"].items()},flush=True)
if __name__=="__main__":main()
