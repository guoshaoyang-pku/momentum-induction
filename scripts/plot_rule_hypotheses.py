"""Plot complete rows only, retain seed counts in labels."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    g=defaultdict(list)
    for f in a.inputs:
        for l in f.open():
            r=json.loads(l);g[r["arm"]].append(r)
    assert all(len({r.get("seed") for r in rs})==len(rs) for rs in g.values())
    arms=[("learned_partition_masked_history","Learned partition"),("train_history_merge","History-only merge"),("known_partition_uniform_prior","Known slot reference")]
    panels=["L3-SC","L3-unfiltered","L1"]
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout="constrained",sharey=True)
    for ax,k in zip(axes,panels):
        for i,(arm,label) in enumerate(arms):
            rs=g[arm]
            if not rs:continue
            a1=[r["eval"][k]["map_seq"]*100 for r in rs]
            a4=[r["eval"][k]["top4_seq"]*100 for r in rs]
            for j,vals in enumerate((a1,a4)):
                sd=np.std(vals,ddof=1) if len(vals)>1 else 0
                ax.bar(i+(j-.5)*.34,np.mean(vals),.32,yerr=sd,label=("MAP, one path","Top 4 path coverage")[j] if i==0 else None,color=("#2563eb","#a0bde9")[j])
        ax.set_xticks(range(3),[f"{label}\n(n={len(g[arm])})" for arm,label in arms],fontsize=8)
        ax.set_title(k);ax.grid(axis="y",alpha=.2);ax.set_ylim(0,105)
    axes[0].set_ylabel("8-step exact (%)")
    axes[0].legend(fontsize=7,frameon=False)
    fig.suptitle("Persistent rule hypotheses: probability-ranked closed-loop futures")
    fig.savefig(a.out,dpi=180)
if __name__=="__main__":main()
