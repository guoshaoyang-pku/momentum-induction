"""Independent final-EMA L2/L3 evaluation for the frozen E22 factorial."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from cawm.train import code_fingerprint, get_corpus
from diag_e12_rope_seam import analyse, load

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dep = json.loads(Path('DEPLOYMENT.json').read_text())
    assert code_fingerprint() == dep['cawm_sha256']
    m, args, step = load(a.run, a.device)
    assert step == 40000 and args['task'] == 'L23'
    assert args['seed'] in (42,43,44) and args['dmodel'] in (64,128)
    assert args['pos_enc'] in ('rope','rope_axial')
    assert args['tokenizer'] in ('cell','pair')
    ckpt = Path('data/ckpt', a.run+'.pt')
    rows = {}
    for label, seed, half in [('l2_val_sc_g8',43,'train'),('l3_zs_sc_g8',44,'zs')]:
        c = get_corpus('data/eval_corpora',label,n=2048,master_seed=seed,
                       half=half,grid=8,self_consistent=True)
        t = time.time()
        torch.manual_seed(0)
        res = analyse(m,c['frames'],a.device,batch=a.batch)
        res['frames_sha256'] = hashlib.sha256(c['frames'].tobytes()).hexdigest()
        res['wall_s'] = time.time()-t
        rows[label] = res
    out = dict(run=a.run,experiment='E22',ckpt_step=step,weights='ema',
               eval_precision='fp32',code=code_fingerprint(),args=args,
               ckpt_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),corpora=rows)
    dest = Path('data/runs',a.run,'e22_eval.json')
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(a.run, {k:v['rollout_seq_acc'] for k,v in rows.items()},flush=True)

if __name__ == '__main__':
    main()
