"""Explicit checkpoint initialization, distinct from optimizer resume."""
import hashlib
from pathlib import Path
import torch


def initialize_and_freeze(model, path, freeze_detectors=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('step') != ck.get('args',{}).get('steps') or 'model_ema' not in ck:
        raise ValueError('Initialization requires final-step EMA checkpoint')
    frozen=[]
    if hasattr(model,'evidence_adapter'):
        missing,unexpected=model.load_state_dict(ck['model_ema'],strict=False)
        expected={k for k in model.state_dict() if k.startswith('evidence_adapter.')}
        if set(missing)!=expected or unexpected:
            raise ValueError('Adapter initialization must match every original weight')
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('evidence_adapter.'))
        frozen=sorted({k.split('.')[0] for k in ck['model_ema']})
    else:
        model.load_state_dict(ck['model_ema'],strict=True)
    if freeze_detectors:
        names=('feat_a','split_a','conjoin_a','feat_b','split_b','conjoin_b') if hasattr(model,'feat_a') else ('feat_s','split_s','conjoin_s')
        for name in names:
            module=getattr(model,name)
            for p in module.parameters():p.requires_grad_(False)
            frozen.append(name)
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                source_seed=ck['args']['seed'],source_steps=ck['step'],source_task=ck['args']['task'],
                weights='model_ema',frozen_modules=frozen,optimizer='fresh',ema='reset_to_loaded_weights')
