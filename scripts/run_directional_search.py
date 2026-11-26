import argparse,json,hashlib
from pathlib import Path
import numpy as np
from cawm.models.directional_search import metric_classes,search,prefix_tables
from run_directional_metric import dataset,SLOTS,digest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n',type=int,default=2048)
    ap.add_argument('--data-seed-base',type=int,default=30000)
    a=ap.parse_args()
    assert not a.out.exists()
    source=Path('results/directional_metric/m2048.jsonl')
    rows=[r for r in map(json.loads,source.open()) if r['model']=='rank2']
    for r in rows:
        seed=r['seed']; x,rules=dataset(a.data_seed_base+seed,a.n)
        train,tr=dataset(20000+seed,r['M'])
        assert digest(train)==r['train_sha'] and not set(map(bytes,tr))&set(map(bytes,rules))
        if a.data_seed_base==30000 and a.n==r['valid_n']: assert digest(x)==r['valid_sha']
        learned=metric_classes(r['weight'])
        same=np.array_equal(learned[:,None]==learned[None,:],SLOTS[:,None]==SLOTS[None,:])
        print(seed,'learned classes',len(np.unique(learned)),'exact partition',same,flush=True)
        dynamic=(x[:,8:]!=x[:,7:8]).any(axis=(1,2,3))
        for name,classes in [('learned_rank2',learned),('known_groups',SLOTS)]:
            pred=search(x[:,:8],classes,seed=seed)
            for k,(p,c,cert) in pred.items():
                eq=p==x[:,8:]; hit=eq.all(axis=(1,2,3))
                result=dict(seed=seed,data_seed=a.data_seed_base+seed,n=a.n,model=name,k=k,classes=int(classes.max()+1),partition_exact=bool(same),valid_sha=digest(x),source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),nonpersistent_n=int(dynamic.sum()),seq8=float(hit.mean()),nonpersistent_seq8=float(hit[dynamic].mean()),pixel8=float(eq.mean()),mean_mass=float(np.exp2(-c.astype(float)).mean()),mean_new_bits=float(c.mean()),map_certified_fraction=float(cert.mean()))
                with a.out.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
                print(result,flush=True)
if __name__=='__main__':main()
