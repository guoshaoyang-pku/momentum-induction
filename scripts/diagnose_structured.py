"""E17 final-checkpoint audit: first errors, evidence strata and retrieval mass."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from cawm.train import build_model, model_kwargs, code_fingerprint
from cawm.eval import rollout_corpus, metrics
from cawm.simulate import neighbor_count, ca_step, rule_tables
from cawm import rules as R


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slots(frames):
    shape=frames.shape
    f=frames.reshape(-1,*shape[-2:])
    return (9*f.astype(np.int16)+neighbor_count(f)).reshape(shape)


def corpus_audit(corpus,l4=False,exercised=True,half='zs'):
    fr,rule,cov=corpus['frames'],corpus['rules'],corpus['covs']
    tables=rule_tables(rule)
    for t in range(15):
        assert np.array_equal(ca_step(fr[:,t],tables),fr[:,t+1])
    ps=slots(fr[:,:7]).reshape(len(fr),-1)
    counts=np.stack([(ps==i).sum(1) for i in range(18)],1)
    assert np.array_equal(counts>0,cov)
    qs=slots(fr[:,7:15]).reshape(len(fr),-1)
    direct=np.take_along_axis(cov,qs,1)
    partner=np.arange(18)
    paired=np.zeros(18,dtype=bool)
    for a,b in R.L4V2_CONFIGS['l4b']['pairs']:
        partner[a],partner[b]=b,a;paired[a]=paired[b]=True
    derived=(~direct)&paired[qs]&np.take_along_axis(cov,partner[qs],1) if l4 else np.zeros_like(direct)
    assert (direct|derived).all(),'unanswerable test cells'
    if l4:
        if exercised:assert derived.any(1).all(),'not exercised corpus'
        assert np.isin(rule,R.l4v2_pool('l4b',half)).all()
    else:
        assert np.isin(rule,R.rule_split()[half=='zs']).all()
    # Infer only from observed transitions; no rule ID used for this oracle.
    observed=np.full((len(fr),18),-1,dtype=np.int16)
    out=fr[:,1:8].reshape(len(fr),-1)
    for slot in range(18):
        hit=ps==slot;seen=hit.any(1)
        one=(out*hit).sum(1)
        assert ((one==0)|(one==hit.sum(1))).all()
        observed[seen,slot]=(one[seen]>0)
    oracle=np.take_along_axis(observed,qs,1)
    oracle[derived]=(1-np.take_along_axis(observed,partner[qs],1))[derived]
    assert np.array_equal(oracle,fr[:,8:].reshape(len(fr),-1))
    return qs,ps,direct,derived,partner,counts


@torch.no_grad()
def audit(checkpoint,out,device='cuda',batch=64,corpus_name=None,sampling_seed=0):
    start=time.time();torch.manual_seed(sampling_seed);np.random.seed(sampling_seed)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);a=ck['args']
    assert ck['step']==a['steps'] and 'model_ema' in ck
    model=build_model(a['seed'],model=a['model'],**model_kwargs(a)).to(device).eval()
    model.load_state_dict(ck['model_ema'],strict=True)
    l4=a.get('l4v2')=='l4b'
    name=corpus_name or ('l4b_zs_scx_g8' if l4 else 'l3_zs_sc_g8')
    cp=Path('data/eval_corpora',name+'.npz')
    with np.load(cp) as z:corpus={k:z[k] for k in z.files}
    qs,ps,direct,derived,partner,counts=corpus_audit(corpus,l4,
        exercised='scx' in name,half='train' if 'val' in name else 'zs')
    # build_model restores CPU RNG but seeds CUDA with the model seed. Keep
    # the historical seed-0 path byte-compatible; reseed AFTER construction
    # for explicitly requested additional sampling passes.
    if sampling_seed != 0:
        torch.manual_seed(sampling_seed)
        np.random.seed(sampling_seed)
    rng_state=(torch.cuda.get_rng_state(device) if device.startswith('cuda')
               else torch.get_rng_state()).cpu().numpy().tobytes()
    rng_sha=hashlib.sha256(rng_state).hexdigest()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    pred=rollout_corpus(model,corpus,device=device,batch=batch,bf16=False)
    target=corpus['frames'][:,8:];eq=pred==target
    exact=eq.all((1,2,3));wrong=(~eq).reshape(len(pred),8,-1)
    first=np.where(exact,-1,wrong.any(2).argmax(1))
    mask=np.zeros_like(wrong)
    for i,t in enumerate(first):
        if t>=0:mask[i,t]=wrong[i,t]
    firstmask=mask.reshape(len(pred),-1)
    cats={'direct':direct,'partner_only':derived,'neither':~(direct|derived)}
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),step=ck['step'],
        args=a,model=a['model'],params=model.total_param_count(),seed=a['seed'],
        weights='model_ema',precision='fp32',sampling_seed=sampling_seed,cawm_sha256=code_fingerprint(),
        rollout_rng_state_sha256=rng_sha,
        sampling_rng_policy=('legacy_seed0_model_constructor_cuda_state' if sampling_seed==0 else 'explicit_reseed_after_model_construction'),
        evaluator_sha256=sha(__file__),hostname=platform.node(),torch_version=str(torch.__version__),
        corpus=name,corpus_sha256=sha(cp),n=len(pred),metrics=metrics(pred,corpus),
        first_error_frame=np.bincount(first[first>=0],minlength=8).tolist(),
        first_error_slot=np.bincount(qs[firstmask],minlength=18).tolist(),
        first_error_category={k:int((firstmask&v).sum()) for k,v in cats.items()},
        first_error_worlds=int((first>=0).sum()),oracle_seq_acc=1.0,
        frame_exact=eq.all((2,3)).mean(0).tolist(),
        corpus_array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in corpus.items()})
    arrays=dict(pred=pred,target=target,exact=exact,first_error_frame=first,
                query_slot=qs,direct=direct,partner_only=derived)
    if not hasattr(model,'denoise'):
        logits=[]
        for i in range(0,len(pred),batch):
            fr=torch.from_numpy(corpus['frames'][i:i+batch]).to(device).float()
            logits.append(model((fr*2-1).unsqueeze(1)).float().cpu().numpy())
        logits=np.concatenate(logits);correct=((logits>0)==target).reshape(len(pred),-1)
        arrays['tf_logit']=logits
        report['tf_pixel_acc']=float(correct.mean())
        report['tf_all_future_exact']=float(correct.all(1).mean())
        report['tf_category']={k:dict(n=int(v.sum()),errors=int((v&~correct).sum()),
            accuracy=float(correct[v].mean()) if v.any() else None) for k,v in cats.items()}
        report['tf_slots']={str(s):dict(n=int((qs==s).sum()),
            errors=int(((qs==s)&~correct).sum()),
            derived_errors=int(((qs==s)&derived&~correct).sum())) for s in range(18)}
        # Attention diagnostics for original1read only. Tokens are labeled by
        # true prefix slots to measure learned cross-slot routing, not assumed
        # to be one-hot or to encode their declared output channel.
        if a['model'] in ('art','art_kvshift') and getattr(model,'reads',1)==1 and (not l4 or 'scx' in name):
            accum={k:dict(n=0,same_mass=0.,partner_mass=0.,entropy=0.) for k in cats}
            for i in range(0,len(pred),16):
                fr=torch.from_numpy(corpus['frames'][i:i+16]).to(device).float()*2-1
                b=len(fr);tok=model._tokens(fr[:,:8].unsqueeze(1))
                wk=model.Wk if isinstance(model.Wk,torch.Tensor) else model.Wk.weight.t()
                k=tok@wk
                for t in range(8):
                    q=model._codes(fr[:,7+t].unsqueeze(1)).flatten(2).transpose(1,2)
                    w=(torch.einsum('bqd,bkd->bqk',q,k)*model.attn_scale).softmax(-1)
                    src=torch.from_numpy(ps[i:i+b]).to(device)
                    dst=torch.from_numpy(qs[i:i+b,t*64:(t+1)*64]).to(device)
                    par=torch.from_numpy(partner).to(device)[dst.long()]
                    same=(w*(src[:,None,:]==dst[:,:,None])).sum(-1).cpu().numpy()
                    pm=(w*(src[:,None,:]==par[:,:,None])*(par!=dst)[:,:,None]).sum(-1).cpu().numpy()
                    ent=(-(w*w.clamp_min(1e-30).log()).sum(-1)).cpu().numpy()
                    for key,v in cats.items():
                        sel=v[i:i+b,t*64:(t+1)*64];acc=accum[key]
                        acc['n']+=int(sel.sum());acc['same_mass']+=float(same[sel].sum())
                        acc['partner_mass']+=float(pm[sel].sum());acc['entropy']+=float(ent[sel].sum())
            report['attention_by_evidence']={key:{k:v/max(acc['n'],1) for k,v in acc.items() if k!='n'}|{'n':acc['n']} for key,acc in accum.items()}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    report['prediction_sha256']=sha(out.with_suffix('.npz'))
    report['wall_time_s']=time.time()-start
    if device.startswith('cuda'):
        report['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
        report['gpu_name']=torch.cuda.get_device_name()
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('checkpoint','metrics','first_error_category','first_error_slot','wall_time_s')},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--corpus',default=None)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=64)
    p.add_argument('--sampling_seed',type=int,default=0)
    a=p.parse_args();audit(a.checkpoint,a.out,a.device,a.batch,a.corpus,a.sampling_seed)


if __name__=='__main__':main()
