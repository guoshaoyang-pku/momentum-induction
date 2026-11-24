"""Training-only repair of overly fragmented learned rule classes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from cawm.models.rule_hypotheses import merge_observed_classes,outer_totalistic_classes
from run_rule_hypotheses import evaluate
from run_raw_evidence_experiment import original,digest
from run_raw_evidence_support import CORPORA

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--corpora",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    corp={}
    for k,(name,sha) in CORPORA.items():
        path=a.corpora/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        corp[k]=np.load(path)["frames"]
    ref=outer_totalistic_classes();true=ref[:,None]==ref[None,:]
    with a.out.open("x") as f:
        for line in a.input.read_text().splitlines():
            r=json.loads(line)
            if "class_map" not in r:continue
            train=original(r["seed"],512);assert digest(train)==r["train_sha"]
            groups,merges=merge_observed_classes(np.array(r["class_map"]),train[:,:8],min_overlap=12)
            eq=groups[:,None]==groups[None,:]
            out=dict(arm="train_history_merge",seed=r["seed"],train_sha=r["train_sha"],min_overlap=12,class_map=groups.tolist(),merges=merges,
                     pair_false_merge=int((eq&~true).sum()),pair_false_split=int((~eq&true).sum()),eval={k:evaluate(x,groups) for k,x in corp.items()})
            f.write(json.dumps(out,sort_keys=True,allow_nan=False)+"\n");f.flush()
            print(out["seed"],int(groups.max())+1,len(merges),out["pair_false_merge"],out["pair_false_split"],{k:(v["map_seq"],v["top4_seq"],v["conflicts"]) for k,v in out["eval"].items()},flush=True)
if __name__=="__main__":
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    main()
