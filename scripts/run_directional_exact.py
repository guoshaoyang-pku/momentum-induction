"""Evaluate exact-search certificates and expected/actual accuracy on frozen splits."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from cawm.models.directional_search import metric_classes,search
from cawm.models.directional_exact import exact_search
from run_directional_metric import dataset,SLOTS,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=2048);p.add_argument('--data-seed-base',type=int,default=40000)
    p.add_argument('--max-nodes',type=int,default=4096)
    a=p.parse_args();assert not a.out.exists()
    source=Path('results/directional_metric/m2048.jsonl')
    for r in map(json.loads,source.open()):
        if r['model']!='rank2':continue
        seed=r['seed'];x,rules=dataset(a.data_seed_base+seed,a.n)
        train,tr=dataset(20000+seed,r['M'])
        assert digest(train)==r['train_sha'] and not set(map(bytes,tr))&set(map(bytes,rules))
        if a.data_seed_base==30000 and a.n==r['valid_n']:assert digest(x)==r['valid_sha']
        c=metric_classes(r['weight'])
        assert np.array_equal(c[:,None]==c[None,:],SLOTS[:,None]==SLOTS[None,:])
        dynamic=(x[:,8:]!=x[:,7:8]).any(axis=(1,2,3))
        t0=time.monotonic()
        pred0,cost0,_=search(x[:,:8],c,(32,128),seed=seed)[32]
        pred,cost,cert,nodes,base=exact_search(x[:,:8],c,seed=seed,max_nodes=a.max_nodes)
        assert np.array_equal(base,cost0) and np.all(cost<=cost0)
        assert np.array_equal(pred[cost==cost0],pred0[cost==cost0])
        def metrics(p,k):
            hit=(p==x[:,8:]).all((1,2,3))
            mass=np.exp2(-k.astype(float))
            return dict(seq8=float(hit.mean()),nonpersistent_seq8=float(hit[dynamic].mean()),
                        mean_mass=float(mass.mean()),nonpersistent_mean_mass=float(mass[dynamic].mean()))
        previous=Path("results/directional_search")/("fresh.jsonl" if a.data_seed_base==40000 else "results.jsonl")
        if a.n==2048 and a.data_seed_base in (30000,40000):
            old=[z for z in map(json.loads,previous.open()) if z["seed"]==seed and z["model"]=="learned_rank2" and z["k"]==32]
            assert len(old)==1 and old[0]["valid_sha"]==digest(x)
            assert all(metrics(pred0,cost0)[k]==old[0][k] for k in ("seq8","nonpersistent_seq8","mean_mass"))
        row=dict(seed=seed,data_seed=a.data_seed_base+seed,n=a.n,valid_sha=digest(x),
                 source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
                 nonpersistent_n=int(dynamic.sum()),baseline=metrics(pred0,cost0),exact=metrics(pred,cost),
                 certified=float(cert.mean()),nonpersistent_certified=float(cert[dynamic].mean()),
                 improved=float((cost<cost0).mean()),nodes_mean=float(nodes.mean()),nodes_max=int(nodes.max()),
                 max_nodes=a.max_nodes,elapsed_s=time.monotonic()-t0)
        with a.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
if __name__=='__main__':main()
