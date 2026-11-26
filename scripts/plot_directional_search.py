import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/directional_search')
fig,ax=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
for file,label in [('results.jsonl','Original validation'),('fresh.jsonl','Fresh held-out rules')]:
    rows=list(map(json.loads,(root/file).open()))
    assert len(rows)==24
    learned={(r['seed'],r['k']):r for r in rows if r['model']=='learned_rank2'}
    known={(r['seed'],r['k']):r for r in rows if r['model']=='known_groups'}
    assert len(learned)==len(known)==12 and learned.keys()==known.keys()
    for key,r in learned.items():
        assert r['partition_exact'] and r['classes']==50
        assert all(r[f]==known[key][f] for f in ('seq8','nonpersistent_seq8','pixel8','mean_mass','mean_new_bits','map_certified_fraction'))
    for i,metric in enumerate(('nonpersistent_seq8','mean_mass')):
        vals=np.array([[learned[s,k][metric] for k in (1,8,32,128)] for s in (42,43,44)])
        ax[i].errorbar(range(4),100*vals.mean(0),yerr=100*vals.std(0,ddof=1),marker='o',capsize=3,label=label)
        print(file,metric,vals.mean(0),vals.std(0,ddof=1))
for a,y in zip(ax,['Nonpersistent SeqAcc8 (%)','Mean posterior path mass (%)']):
    a.set_xticks(range(4),['1','8','32','128']);a.set_xlabel('Rule hypotheses');a.set_ylabel(y)
    a.grid(alpha=.2);a.legend(frameon=False,fontsize=8)
fig.savefig(root/'comparison.png',dpi=200)
