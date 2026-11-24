"""Reproduce the held-out noisy CA diagnostics from small JSONL receipts."""
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    a=argparse.ArgumentParser();a.add_argument("--input",type=Path,required=True);a.add_argument("--out",type=Path,required=True);a=a.parse_args()
    r=[json.loads(l) for l in a.input.read_text().splitlines()]
    ps=sorted({x["true_p"] for x in r})
    arms=("learned","known_noise","wrong_center","fair_coin")
    assert len(r)==len(ps)*len(arms)*3==72
    assert len({(x["seed"],x["true_p"],x["arm"]) for x in r})==72
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for arm in arms:
        g=[[x for x in r if x["arm"]==arm and x["true_p"]==p] for p in ps]
        for ax,fn in [(axs[0],lambda x:x["estimated_p"]),(axs[1],lambda x:x["joint_nll_by_frame"][0]),(axs[2],lambda x:np.mean(x["joint_nll_by_frame"]))]:
            y=np.array([[fn(x) for x in a] for a in g])
            ax.errorbar(ps,y.mean(1),yerr=y.std(1,ddof=1),marker="o",ms=3,label=arm)
    axs[0].plot(ps,ps,"k:",label="true rate")
    h=[0 if p==0 else -p*math.log(p)-(1-p)*math.log(1-p) for p in ps]
    for ax in axs[1:]:ax.plot(ps,h,"k:",label="known-rule entropy")
    for ax,t in zip(axs,("Noise fitted from train prefixes","Frame 8 joint NLL / bit","8-frame joint NLL / bit")):
        ax.set(xlabel="True independent flip probability",title=t);ax.grid(alpha=.2)
    axs[0].set_ylabel("Estimated flip rate")
    for ax in axs[1:]:ax.set_ylabel("nats / binary target")
    axs[1].legend(fontsize=7,frameon=False)
    fig.savefig(a.out,dpi=180)
if __name__=="__main__":main()
