"""Plot frozen-target history ablation, with SD across independent seeds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path('results/directional_history')
    rows=[json.loads(line) for line in (root/'fresh.jsonl').read_text().splitlines()]
    hs=[1,2,4,8]
    assert len(rows)==12 and {(r['seed'],r['history']) for r in rows}=={(s,h) for s in (42,43,44) for h in hs}
    for seed in (42,43,44):
        assert len({r['valid_sha'] for r in rows if r['seed']==seed})==1
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.1),layout='constrained')
    def line(ax,key,label,scale=1):
        a=np.array([[next(r[key] for r in rows if r['seed']==s and r['history']==h) for h in hs] for s in (42,43,44)])*scale
        ax.errorbar(hs,a.mean(0),yerr=a.std(0,ddof=1),marker='o',capsize=3,label=label)
        return a.mean(0).tolist()
    summary={}
    summary['known_slots']=line(axes[0],'known_slots','Observed slots')
    axes[0].set(ylabel='Observed rule outputs / 50',ylim=(-2,52))
    summary['first_joint_nll_bits']=line(axes[1],'first_joint_nll_bits','Frame 8')
    summary['trajectory_nll_bits']=line(axes[1],'trajectory_nll_bits','Frames 8–15')
    axes[1].set(ylabel='Joint NLL (bits / world)')
    axes[1].legend(frameon=False,fontsize=8)
    for k,label in [('seq8','Measured SeqAcc8'),('map_lower','Selected model mass'),('map_upper','First-frame MAP bound')]:
        summary[k]=line(axes[2],k,label,100)
    axes[2].set(ylabel='Accuracy or model mass (%)',ylim=(0,100))
    axes[2].legend(frameon=False,fontsize=7)
    for ax in axes:
        ax.set(xlabel='Observed suffix frames',xticks=hs)
        ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):
        fig.savefig(root/('comparison.'+ext),dpi=180)
    (root/'summary.json').write_text(json.dumps({'history':hs,'means':summary,'seeds':[42,43,44],'n_per_seed':2048},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
