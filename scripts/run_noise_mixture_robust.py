"""Prespecified 10% uniform floor on previously fitted population priors."""
import argparse,json
from pathlib import Path
import numpy as np
from cawm.models import noise_mixture as mix
from run_noise_mixture import corpus,stats
from run_raw_evidence_experiment import digest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--maps",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    maps={r["seed"]:np.asarray(r["class_map"],dtype=np.int16) for r in map(json.loads,a.maps.read_text().splitlines())}
    rows=[r for r in map(json.loads,a.input.read_text().splitlines()) if r["arm"]=="learned_prior"]
    with a.out.open("x") as f:
        for r in rows:
            c=maps[r["seed"]]; assert digest(c)==r["map_sha"]
            rates=[0,.03,.1,.25,.5] if r["split"]=="matched" else [.005,.047,.137,.263,.477]
            va,q=corpus(r["seed"],"zs",512,rates,[.15,.15,.35,.2,.15])
            assert digest(va)==r["eval_sha"]
            g=np.array(r["grid"]);w=.9*np.array(r["prior"])+.1/len(g)
            post,_=mix.posterior(va[:,:8],c,g,w)
            r.update(arm="floor_10pct",prior=w.tolist(),rate_mae=float(abs((post*g).sum(1)-q).mean()),**stats(mix.sequence_scores(va,c,g,w)))
            f.write(json.dumps(r,allow_nan=False)+"\n")
if __name__=="__main__":main()
