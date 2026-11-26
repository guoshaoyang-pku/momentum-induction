"""Fixed target, varying observed history, independent fair-rule directional CA.

The decoder receives only the observed suffix. True trajectory likelihood is
computed separately for scoring, not for candidate selection.
"""
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np
from cawm.models.directional_metric import patterns
from cawm.models.directional_search import canonical_classes, metric_classes, prefix_tables, search
from run_directional_metric import dataset, digest

def likelihood(prefix, future, classes):
    """Exact joint NLL in bits: distinct newly observed rule outputs."""
    classes=canonical_classes(classes)
    if future.ndim!=4 or future.shape[0]!=len(prefix) or future.shape[2:]!=prefix.shape[2:] or not np.isin(future,[0,1]).all():
        raise ValueError('invalid future')
    t=prefix_tables(prefix,classes,history=prefix.shape[1])
    row=np.arange(len(prefix))[:,None]
    cur=prefix[:,-1]
    costs=[]; unknown_pixels=[]
    for nxt in future.transpose(1,0,2,3):
        q=classes[patterns(cur)].reshape(len(t),-1)
        y=nxt.reshape(len(t),-1)
        u=t[row,q]<0
        if np.any((~u)&(t[row,q]!=y)):raise ValueError('inconsistent future')
        before=(t>=0).sum(1)
        t[row,q]=y
        if not np.array_equal(t[row,q],y):raise ValueError('inconsistent future')
        costs.append((t>=0).sum(1)-before)
        unknown_pixels.append(u.sum(1))
        cur=nxt
    return np.stack(costs,1),np.stack(unknown_pixels,1)

def evaluate(prefix,future,classes,seed):
    classes=canonical_classes(classes)
    hist=prefix.shape[1]
    known=(prefix_tables(prefix,classes,history=hist)>=0).sum(1)
    pred,cost,_=search(prefix,classes,(32,128),seed=seed,horizon=future.shape[1],history=hist)[32]
    bits,unknown=likelihood(prefix,future,classes)
    # Lower bound on optimal trajectory MAP mass: mass of selected candidate.
    # Upper bound: MAP mass of first complete frame (same for every outcome).
    lower=np.exp2(-cost.astype(float))
    upper=np.exp2(-bits[:,0].astype(float))
    assert np.all(lower<=upper)
    hit=(pred==future).all((1,2,3))
    return dict(history=hist,n=len(prefix),known_slots=float(known.mean()),
        unknown_first_pixels=float(unknown[:,0].mean()),
        first_joint_nll_bits=float(bits[:,0].mean()),
        first_parallel_bce=float(unknown[:,0].mean()*math.log(2)/64),
        trajectory_nll_bits=float(bits.sum(1).mean()),
        seq8=float(hit.mean()),map_lower=float(lower.mean()),map_upper=float(upper.mean()),
        exact_probability_gap=float((upper-lower).mean()),
        bound_equal_fraction=float((upper==lower).mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=2048);p.add_argument('--data-seed-base',type=int,default=40000)
    a=p.parse_args()
    source=Path('results/directional_metric/m2048.jsonl')
    with a.out.open('x') as f:
        for r in map(json.loads,source.open()):
            if r['model']!='rank2':continue
            seed=r['seed'];x,rules=dataset(a.data_seed_base+seed,a.n)
            train,tr=dataset(20000+seed,r['M'])
            assert digest(train)==r['train_sha'] and not set(map(bytes,tr))&set(map(bytes,rules))
            c=metric_classes(r['weight'])
            prev=None
            for h in (1,2,4,8):
                row=evaluate(x[:,8-h:8],x[:,8:],c,seed)
                if prev:
                    assert row['known_slots']>=prev['known_slots'] and row['trajectory_nll_bits']<=prev['trajectory_nll_bits']
                prev=row
                if h==8 and a.n==2048 and a.data_seed_base in (30000,40000):
                    previous=Path('results/directional_search')/('fresh.jsonl' if a.data_seed_base==40000 else 'results.jsonl')
                    old=[z for z in map(json.loads,previous.open()) if z['seed']==seed and z['model']=='learned_rank2' and z['k']==32]
                    assert len(old)==1 and old[0]['valid_sha']==digest(x)
                    assert old[0]['seq8']==row['seq8'] and old[0]['mean_mass']==row['map_lower']
                row.update(seed=seed,data_seed=a.data_seed_base+seed,valid_sha=digest(x),source_sha=hashlib.sha256(source.read_bytes()).hexdigest())
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();print(json.dumps(row),flush=True)
if __name__=='__main__':main()
