"""Plot exact MAP certification and compare the unchanged prior with realized accuracy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("results/directional_exact")
groups=[[json.loads(s) for s in (root/f"{name}.jsonl").open()] for name in ("original","fresh")]
assert all(len(g)==3 and {r["seed"] for r in g}=={42,43,44} for g in groups)
fig,axes=plt.subplots(1,2,figsize=(8.4,3.4))
for ax,rows,title in zip(axes,groups,("Original validation","Fresh validation")):
    for i,metric in enumerate(("seq8","mean_mass")):
        for j,method in enumerate(("baseline","exact")):
            values=100*np.array([r[method][metric] for r in rows])
            x=i+(j-.5)*.30
            ax.bar(x,values.mean(),width=.28,color=("#6685a7","#ce8956")[j],label=("32 candidates","Certified MAP")[j] if i==0 else None)
            ax.scatter(x+np.linspace(-.035,.035,len(values)),values,s=14,c="black",zorder=3)
    ax.set_xticks([0,1],["8-step SeqAcc","Model probability"])
    ax.set_ylim(0,42)
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.spines[["right","top"]].set_visible(False)
    ax.text(.03,.96,"100% MAP certified",transform=ax.transAxes,va="top",fontsize=9)
axes[0].legend(frameon=False,fontsize=8,loc="lower left")
fig.tight_layout()
fig.savefig(root/"comparison.png",dpi=200)
plt.close(fig)
for name,rows in zip(("original","fresh"),groups):
    print(name)
    for key in ("seq8","nonpersistent_seq8","mean_mass"):
        print(key,*[round(100*np.mean([r[m][key] for r in rows]),5) for m in ("baseline","exact")])
    print("certified, improved, mean nodes, max nodes",np.mean([r["certified"] for r in rows]),np.mean([r["improved"] for r in rows]),np.mean([r["nodes_mean"] for r in rows]),max(r["nodes_max"] for r in rows))
