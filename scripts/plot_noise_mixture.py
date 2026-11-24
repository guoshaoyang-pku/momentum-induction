import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument("--dir",type=Path,required=True);a=p.parse_args()
    rows=[r for f in ("results.jsonl","robust.jsonl") for r in map(json.loads,(a.dir/f).read_text().splitlines())]
    arms=["global","uniform_grid","learned_prior","floor_10pct","known_rate"]
    fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for j,split in enumerate(("matched","off_grid")):
        means=[];sd=[]
        for arm in arms:
            v=[r["eight_nll"] for r in rows if r["split"]==split and r["arm"]==arm]
            assert len(v)==3
            means.append(np.mean(v));sd.append(np.std(v,ddof=1))
        ax[j].bar(range(5),means,yerr=sd)
        ax[j].set(xticks=range(5),xticklabels=["Global","Uniform","Learned","10% floor","Known rate"],title=split,ylabel="8-frame joint NLL / bit (nats)",ylim=(0,.55))
        ax[j].tick_params(axis="x",labelrotation=20)
    fig.savefig(a.dir/"comparison.png",dpi=160)
    for split in ("matched","off_grid"):
        for arm in arms:
            v=[r for r in rows if r["split"]==split and r["arm"]==arm]
            print(split,arm,*[round(np.mean([r[k] for r in v]),6) for k in ("first_nll","eight_nll")],None if v[0]["rate_mae"] is None else round(np.mean([r["rate_mae"] for r in v]),6))
if __name__=="__main__":main()
