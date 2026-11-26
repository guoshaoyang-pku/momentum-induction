"""Plot seed means and sample standard deviations; no Structance fit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'results/directional_metric'
rows=[json.loads(l) for p in sorted(root.glob('*.jsonl')) for l in p.read_text().splitlines()]
assert len(rows)==42
assert len({(r['M'],r['model'],r['seed']) for r in rows})==len(rows)
for seed in (42,43,44):
    valid={r['valid_sha'] for r in rows if r['seed']==seed}
    assert len(valid)==1
    for m in (128,512,2048):
        rr=[r for r in rows if r['seed']==seed and r['M']==m]
        assert len({r['train_sha'] for r in rr})==1
        assert all(r['N']==64*m and r['valid_n']==2048 for r in rr)
for r in rows:
    assert np.isfinite(r['train_bce']) and np.isfinite(r['valid_bce'])
    if r['model']!='exact_group_oracle':
        assert r['trace'][-1]['train_bce']==r['train_bce']
        assert not r['certified_converged']
def vals(m,model,metric):
    rr=[r for r in rows if r['M']==m and r['model']==model]
    assert len(rr)==3
    return np.asarray([r['valid_bce'] if metric=='bce' else 100*r['rollout']['nonpersistent_seq8'] for r in rr])
models=['exact_group_oracle','frozen_R0_geometry','rank1','rank2','rank3','rank3_shift']
labels=['Known groups','Frozen metric','Rank 1','Rank 2','Rank 3','Shifted rank 3']
fig,ax=plt.subplots(2,2,figsize=(9,6),layout='constrained')
for j,metric in enumerate(['bce','seq']):
    for i,(model,label) in enumerate(zip(models,labels)):
        v=vals(512,model,metric)
        ax[0,j].bar(i,v.mean(),yerr=v.std(ddof=1),capsize=3)
        if model in ('exact_group_oracle','rank1','rank2','rank3'):
            vv=np.stack([vals(m,model,metric) for m in (128,512,2048)])
            ax[1,j].errorbar([128,512,2048],vv.mean(1),yerr=vv.std(1,ddof=1),marker='o',label=label,capsize=3)
    ax[0,j].set_xticks(range(len(labels)),labels,rotation=25,ha='right')
    ax[0,j].set_title('M=512, 400 updates')
    ax[1,j].set_xscale('log',base=2);ax[1,j].set_xticks([128,512,2048],['128','512','2048'])
    ax[1,j].set_xlabel('Training worlds M');ax[1,j].set_title('Finite budgets: 800 / 400 / 800 updates')
    for a in ax[:,j]:a.set_ylabel('Validation BCE (nats/bit)' if j==0 else 'Nonpersistent SeqAcc8 (%)')
ax[1,0].legend(fontsize=8)
fig.savefig(root/'comparison.pdf')
for m in (128,512,2048):
    for model in models:
        if not any(r['M']==m and r['model']==model for r in rows):continue
        v=vals(m,model,'bce');s=vals(m,model,'seq')
        print(m,model, f'{v.mean():.5f} +/- {v.std(ddof=1):.5f}', f'{s.mean():.2f} +/- {s.std(ddof=1):.2f}')
# Metric separation diagnostics using true groups only for post-hoc evaluation.
raw=((np.arange(512)[:,None]>>np.arange(9))&1)*2-1
bits=(raw+1)//2
slots=25*bits[:,4]+5*bits[:,[1,3,5,7]].sum(1)+bits[:,[0,2,6,8]].sum(1)
same=(slots[:,None]==slots[None,:]) & ~np.eye(512,dtype=bool)
diff=slots[:,None]!=slots[None,:]
for model in ['frozen_R0_geometry','rank1','rank2','rank3']:
    within=[]; between=[]
    for r in rows:
        if r['M']!=512 or r['model']!=model:continue
        a=raw@np.asarray(r['weight']).T
        k=np.exp(-((a[:,None]-a[None,:])**2).sum(-1))
        within.append(k[same].mean());between.append(k[diff].mean())
    print('geometry',model,np.mean(within),np.mean(between))
