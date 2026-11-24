"""Plot three-seed closed-loop and full-corpus first-frame BCE."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rows",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.rows.read_text().splitlines()]
    assert len(rows)==9 and len({(r["mode"],r["seed"]) for r in rows})==9
    modes=("mean","count","sqrt");corpora=("L3-SC","L3-unfiltered","L1")
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.2),layout="constrained")
    for i,mode in enumerate(modes):
        r=[v for v in rows if v["mode"]==mode]
        x=np.arange(3)+(i-1)*.24
        for ax,metric in zip(axs[:2],("seq8","bce")):
            y=np.array([[v["eval"][k][metric] for k in corpora] for v in r])
            ax.bar(x,y.mean(0),.23,yerr=y.std(0,ddof=1),label=mode,capsize=3)
            ax.scatter(np.tile(x,(3,1)),y,s=8,color="black",zorder=4)
        y=np.array([[v["eval"][k]["incomplete"]["seq8"] for k in corpora[1:]] for v in r])
        axs[2].bar(x[:2],y.mean(0),.23,yerr=y.std(0,ddof=1),label=mode,capsize=3)
        axs[2].scatter(np.tile(x[:2],(3,1)),y,s=8,color="black",zorder=4)
    for ax,title in zip(axs,("All worlds: 8-step SeqAcc","Full-corpus first-frame BCE","Unseen rule entries: 8-step SeqAcc")):
        ax.set_title(title,fontsize=10);ax.grid(axis="y",alpha=.2)
    for ax in axs[:2]:ax.set_xticks(range(3),corpora,rotation=15)
    axs[2].set_xticks(range(2),corpora[1:],rotation=15)
    for ax in (axs[0],axs[2]):ax.set_ylim(0,1.25)
    axs[1].set_yscale("log");axs[1].set_ylim(.0002,.04)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Original L3-SC training | M=512, 2400 updates, seeds 42/43/44")
    fig.savefig(a.out,dpi=160)
if __name__=="__main__":main()
