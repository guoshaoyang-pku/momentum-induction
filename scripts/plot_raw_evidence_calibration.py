"""Matched 3-seed comparisons; distinguish source distribution and eval cohort."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--new",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(l) for l in a.new.read_text().splitlines()]
    base=[dict(json.loads(l),distribution="L3-SC") for l in a.baseline.read_text().splitlines()]
    rows+= [r for r in base if r["mode"] in ("mean","count")]
    dist=("L3-SC","L3-unfiltered")
    modes=("mean","count","support")
    g={(d,m):sorted([r for r in rows if (r["distribution"],r["mode"])==(d,m)],key=lambda r:r["seed"]) for d in dist for m in modes}
    for d in dist:
        assert all([r["seed"] for r in g[d,m]]==[42,43,44] for m in modes)
        assert len({tuple(r["train_sha"] for r in g[d,m]) for m in modes})==1
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    metrics=(("L3-SC","seq8",None,"L3-SC: eight-step exact (%)"),("L3-unfiltered","seq8",None,"Unfiltered L3: eight-step exact (%)"),("L1","seq8",None,"L1: eight-step exact (%)"),("L3-unfiltered","unseen_slot","bce","Unfiltered L3: unseen-slot first-frame BCE"))
    for ax,(ev,metric,sub,title) in zip(axs.flat,metrics):
        for j,d in enumerate(dist):
            vals=np.array([[r["eval"][ev][metric] if sub is None else r["eval"][ev][metric][sub] for r in g[d,m]] for m in modes])
            if sub is None:vals*=100
            x=np.arange(3)+(j-.5)*.36
            ax.bar(x,vals.mean(1),.32,yerr=vals.std(1,ddof=1),label="train "+d,alpha=.75)
            for i,xx in enumerate(x):
                ax.scatter(xx+np.array([-.055,0,.055]),vals[i],s=13,color="black",zorder=4)
        ax.set(xticks=np.arange(3),xticklabels=modes,title=title)
        if sub is None:ax.set_ylim(0,110)
        else:ax.axhline(np.log(2),ls="--",lw=1,color="gray",label="fair coin")
        ax.grid(axis="y",alpha=.18);ax.legend(fontsize=8,frameon=False)
    fig.suptitle("Same 512 trajectories per seed and source distribution; 2,400 steps")
    fig.savefig(a.out,dpi=170);plt.close(fig)
    for d in dist:
        for m in modes:
            r=g[d,m]
            vals=[]
            for ev,metric,sub,_ in metrics:
                v=np.array([z["eval"][ev][metric] if sub is None else z["eval"][ev][metric][sub] for z in r])
                vals.append([float(v.mean()),float(v.std(ddof=1))])
            print(json.dumps(dict(distribution=d,mode=m,metrics=vals,train_bce=float(np.mean([z["train_bce"] for z in r])),fully_covered=[z["eval"]["L3-unfiltered"]["fully_covered"]["seq8"] for z in r],incomplete=[z["eval"]["L3-unfiltered"]["incomplete"]["seq8"] for z in r])))
if __name__=="__main__":main()
