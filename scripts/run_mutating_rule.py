"""Independent held-out-rule experiment: rule mutation versus pixel flips."""
import argparse,json
from pathlib import Path
import numpy as np
from cawm.models import mutating_rule as m
from cawm.data import _draw_sample_params
from cawm.rules import rule_split
from cawm.simulate import ca_step,rule_tables
from run_raw_evidence_experiment import digest

def corpus(seed,half,n,p,q):
    pool=rule_split()[0 if half=="train" else 1]
    a=[_draw_sample_params(i,1000+seed+(0 if half=="train" else 10000),pool,8) for i in range(n)]
    ids,init=zip(*a)
    rng=np.random.default_rng([20261125,seed,0 if half=="train" else 1,int(p*10000),int(q*10000)])
    tab=rule_tables(ids);x=np.empty((n,16,8,8),np.uint8);x[:,0]=np.stack(init)
    for t in range(15):
        if t:tab^=(rng.random(tab.shape)<q).astype(np.uint8)
        x[:,t+1]=ca_step(x[:,t],tab)^(rng.random((n,8,8))<p).astype(np.uint8)
    return x

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--maps",type=Path,required=True)
    a.add_argument("--off-grid",action="store_true")
    a=a.parse_args()
    maps={r["seed"]:np.asarray(r["class_map"],dtype=np.int16) for r in map(json.loads,a.maps.read_text().splitlines())}
    assert set(maps)=={42,43,44}
    grid=[0,.005,.01,.025,.05,.075,.1,.15,.2,.3,.4,.5]
    candidates=[(p,q) for p in grid for q in grid]
    static=[(p,0) for p in grid]
    with a.out.open("x") as f:
        for seed,c in sorted(maps.items()):
            assert len(np.unique(c))==18
            cases=[(.037,.137)] if a.off_grid else [(0,0),(.1,0),(0,.1),(.05,.1),(.1,.3),(.5,.1)]
            for p,q in cases:
                tr=corpus(seed,"train",512,p,q);va=corpus(seed,"zs",256,p,q)
                r,ll=m.fit_rates(tr[:,:8],c,candidates)
                s,sl=m.fit_rates(tr[:,:8],c,static)
                for arm,rates in [("dynamic",r),("static",s),("known_rates",(p,q))]:
                    score=m.sequence_scores(va,c,*rates)
                    if not np.isfinite(score).all():
                        # Explicit failure, never replace infinity by a finite NLL.
                        first=eight=None
                    else:first=float(score[:,0].mean()/64);eight=float(score.mean()/64)
                    y=m.rollout(va[:,:8],c,*rates,seed=seed)
                    hit=y==va[:,8:];exact=hit.reshape(len(y),8,-1).all(2).cumprod(1)
                    row=dict(seed=seed,p=p,q=q,arm=arm,fit_p=float(rates[0]),fit_q=float(rates[1]),train_M=len(tr),eval_M=len(va),train_sha=digest(tr[:,:8]),eval_sha=digest(va),map_sha=digest(c),first_nll=first,eight_nll=eight,invalid_conditionals=int((~np.isfinite(score)).sum()),sample_pixel=float(hit.mean()),sample_cumulative_exact=exact.mean(0).tolist())
                    f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
                print(seed,p,q,r,s,flush=True)
if __name__=="__main__":main()
