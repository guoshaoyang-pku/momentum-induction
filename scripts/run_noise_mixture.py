"""Fit and test trajectory-level noise heterogeneity on held-out rules."""
import argparse,json
from pathlib import Path
import numpy as np
from cawm.models import noise_mixture as mix, noisy_posterior as single
from run_raw_evidence_experiment import noisy,digest

def corpus(seed,half,n,rates,weights):
    rng=np.random.default_rng([20261125,seed,0 if half=="train" else 1])
    p=rng.choice(rates,n,p=weights)
    x=np.empty((n,16,8,8),np.uint8)
    for q in rates:
        take=p==q
        # Same rule/initial-condition stream regardless of assigned noise.
        x[take]=noisy(seed,q,half,n)[take]
    return x,p

def stats(loss):
    return dict(first_nll=float(loss[:,0].mean()/64),eight_nll=float(loss.mean()/64))

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--maps",type=Path,required=True)
    a=a.parse_args()
    maps={r["seed"]:np.asarray(r["class_map"],dtype=np.int16) for r in map(json.loads,a.maps.read_text().splitlines())}
    grid=np.linspace(0,.5,51)
    train_rates=[0,.03,.1,.25,.5];weights=[.15,.15,.35,.2,.15]
    with a.out.open("x") as f:
        for seed,c in sorted(maps.items()):
            assert len(np.unique(c))==18
            tr,p=corpus(seed,"train",1024,train_rates,weights)
            fit,trace=mix.fit_prior(tr[:,:8],c,grid)
            glob=single.fit_noise(tr[:,:8],c)
            for name,rates in [("matched",train_rates),("off_grid",[.005,.047,.137,.263,.477])]:
                va,q=corpus(seed,"zs",512,rates,weights)
                # A known-rate comparator still infers latent rule bits.
                oracle=np.empty((len(va),8))
                for v in rates:
                    mask=q==v
                    if mask.any():oracle[mask]=single.sequence_scores(va[mask],c,v)[0]
                post,_=mix.posterior(va[:,:8],c,grid,fit)
                # Learned prior may assign tiny/zero weight to unseen rates;
                # retain as-is to expose distribution shift.
                preds=(post*grid).sum(1)
                for arm,loss in [
                    ("global",single.sequence_scores(va,c,glob)[0]),
                    ("uniform_grid",mix.sequence_scores(va,c,grid,np.full(len(grid),1/len(grid)))),
                    ("learned_prior",mix.sequence_scores(va,c,grid,fit)),
                    ("known_rate",oracle)]:
                    row=dict(seed=seed,split=name,arm=arm,train_M=len(tr),eval_M=len(va),
                        train_sha=digest(tr[:,:8]),eval_sha=digest(va),map_sha=digest(c),
                        global_p=glob,grid=grid.tolist(),prior=fit.tolist(),em_iterations=len(trace)-1,
                        em_improvement=trace[-1]-trace[0],rate_mae=float(abs(preds-q).mean()) if arm=="learned_prior" else None,
                        **stats(loss))
                    f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
                print(seed,name,"global",glob,"MAE",float(abs(preds-q).mean()),flush=True)
if __name__=="__main__":main()
