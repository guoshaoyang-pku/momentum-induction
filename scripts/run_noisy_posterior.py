"""Train-prefix noise MLE, exact predictive NLL and true closed-loop samples."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
from cawm.models.noisy_posterior import fit_noise,sequence_scores,rollout
from run_raw_evidence_experiment import noisy,digest

def summarize(x,c,p,seed):
    j,m,pixel=sequence_scores(x,c,p)
    h,w=x.shape[-2:]
    pred=rollout(x[:,:8],c,p,seed,False)
    draw=rollout(x[:,:8],c,p,seed,True)
    def metric(y):
        hit=y==x[:,8:]
        return dict(pixel=hit.mean(axis=(0,2,3)).tolist(),seq=np.logical_and.accumulate(hit.all(axis=(2,3)),axis=1).mean(0).tolist())
    def loss(a):
        return [float(v) if np.isfinite(v) else None for v in a.mean(0)/(h*w)]
    return dict(joint_nll_by_frame=loss(j),pixel_marginal_nll_by_frame=loss(m),
                impossible_sequences=int((~np.isfinite(j)).any(1).sum()),teacher_forced_pixel=pixel.mean(0).tolist(),
                greedy=metric(pred),sample=metric(draw))

def main():
    p=argparse.ArgumentParser();p.add_argument("--maps",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    maps={r["seed"]:np.asarray(r["class_map"],dtype=np.int16) for r in map(json.loads,a.maps.read_text().splitlines())}
    assert set(maps)=={42,43,44}
    with a.out.open("x") as f:
        for seed,c in sorted(maps.items()):
            assert len(np.unique(c))==18
            # Canonicalization is irrelevant to prediction; preserve frozen map.
            ids=np.arange(512);xor=((ids>>0)^(ids>>4))&1
            bad=c[ids^(xor|(xor<<4))]
            for p in (0.,.01,.05,.1,.2,.5):
                tr=noisy(seed,p,"train",2048)[:512,:8]
                va=noisy(seed,p,"zs",256)
                q=fit_noise(tr,c)
                # A wrong-neighborhood control independently fits its own noise rate.
                qbad=fit_noise(tr,bad)
                arms={"learned":(c,q),"known_noise":(c,p),"wrong_center":(bad,qbad),"fair_coin":(c,.5)}
                for arm,(classes,eps) in arms.items():
                    # A zero-rate model can assign zero mass to a noisy prefix;
                    # record that case explicitly rather than clipping scores.
                    try:r=summarize(va,classes,eps,seed+1234)
                    except ValueError as e:
                        if str(e)!="impossible prefix":raise
                        r={"error":str(e)}
                    row=dict(seed=seed,true_p=p,arm=arm,estimated_p=eps,train_M=512,eval_M=256,
                        train_prefix_sha=digest(tr),eval_sha=digest(va),classes_sha=digest(classes),distribution="unfiltered-noise",**r)
                    f.write(json.dumps(row,allow_nan=False,sort_keys=True)+"\n");f.flush()
                print(seed,p,q,qbad,flush=True)
if __name__=="__main__":main()
