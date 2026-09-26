import torch
import pytest
from cawm.train import build_model,EMA
from cawm.initialization import initialize_and_freeze

def test_init_final_ema_and_detector_freeze(tmp_path):
    for model in ('art','art_kvshift'):
        src=build_model(42,model=model,grid=8,arm='emergence',head='concat')
        p=tmp_path/(model+'.pt');torch.save(dict(step=40,args=dict(steps=40,seed=42,task='L23'),model_ema=src.state_dict()),p)
        dst=build_model(43,model=model,grid=8,arm='emergence',head='concat')
        info=initialize_and_freeze(dst,p,True)
        assert info['source_seed']==42 and len(info['sha256'])==64
        assert all(torch.equal(v,dst.state_dict()[k]) for k,v in src.state_dict().items())
        assert dst.Wk.weight.requires_grad and dst.head[0].weight.requires_grad
        for name in info['frozen_modules']:assert all(not p.requires_grad for p in getattr(dst,name).parameters())
        ema=EMA(dst,.999)
        assert all(torch.equal(v,dst.state_dict()[k]) for k,v in ema.shadow.items())
        # Real gradient step cannot modify the frozen local representation.
        frozen={k:v.clone() for k,v in dst.state_dict().items() if k.split('.')[0] in info['frozen_modules']}
        x=torch.randint(0,2,(2,1,16,8,8)).float()*2-1
        dst(x).sum().backward();opt=torch.optim.Adam([p for p in dst.parameters() if p.requires_grad]);opt.step()
        assert all(torch.equal(v,dst.state_dict()[k]) for k,v in frozen.items())

def test_reject_nonfinal_or_nonema(tmp_path):
    m=build_model(42,model='art',grid=8,arm='emergence',head='concat');p=tmp_path/'bad.pt'
    torch.save(dict(step=10,args=dict(steps=40),model_ema=m.state_dict()),p)
    with pytest.raises(ValueError):initialize_and_freeze(m,p)
