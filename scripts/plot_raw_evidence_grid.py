"""Plot seed means and standard deviations of held-out autonomous rollouts."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rows=[json.loads(l) for l in a.input.open()]
    assert len(rows)==6 and {(r["arm"],r["seed"]) for r in rows}=={(a,s) for a in ("grid8","mixed") for s in (42,43,44)}
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout="constrained",sharey=True)
    for g,ax in zip(("4","8","16"),axs):
        for arm in ("grid8","mixed"):
            rr=[r for r in rows if r["arm"]==arm]
            x=np.array([r["long"][g]["cumulative_exact"] for r in rr])*100
            mean=x.mean(0);sd=x.std(0,ddof=1)
            ax.plot(np.arange(1,33),mean,label=arm)
            ax.fill_between(np.arange(1,33),mean-sd,mean+sd,alpha=.15)
            sc=np.array([r["eval"][g]["seq8"] for r in rr])
            cov=np.array([r["long"][g]["covered_seq32"] for r in rr])
            assert len({r["long"][g]["sha"] for r in rows})==1
            print(g,arm,"SC8",sc.mean(),"SC BCE",np.mean([r["eval"][g]["bce"] for r in rr]),"unfiltered8/32",mean[7],mean[-1],"32sd",sd[-1],"covered n",rr[0]["long"][g]["covered_n"],"covered32",cov.mean())
        ax.set(title=f"{g} × {g}",xlabel="Autonomous future steps",ylim=(0,102));ax.grid(alpha=.2);ax.legend()
    axs[0].set_ylabel("Cumulative exact trajectory (%)")
    fig.suptitle("Unfiltered unseen rules: cross-grid transfer to 32-step rollout")
    fig.savefig(a.out,dpi=180)
if __name__=="__main__":main()
