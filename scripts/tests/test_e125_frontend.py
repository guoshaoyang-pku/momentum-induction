import torch
from cawm.models.diffusion import DiffusionModel
from cawm.train import build_model,model_kwargs

def make(arm='emergence',**kw):
    torch.manual_seed(42)
    return DiffusionModel(arm=arm,**kw)

def test_default_preserved_and_common_initialization_matched():
    a,b=make(),make(cond_pairing='conv')
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
    c=make(cond_pairing='kvshift')
    for k,v in a.state_dict().items():
        if not k.startswith(('feat_a.','split_a.','conjoin_a.')):
            assert torch.equal(v,c.state_dict()[k]),k

def test_analytic_condition_matches_conv_exactly():
    a,b=make('existence'),make('existence',cond_pairing='kvshift')
    # Compare algebra in float64: platform-specific fp32 convolution summation
    # differs by ~1e-6 on Apple CPU, even for analytically identical features.
    a,b=a.double(),b.double()
    x=torch.randint(0,2,(8,1,8,8,8)).double()*2-1
    assert torch.equal(a.forward_a(x),b.forward_a(x))

def test_train_reload_and_frontend_gradients():
    kw=model_kwargs(dict(grid=8,arm='emergence',head='concat',cond_pairing='kvshift'))
    a=build_model(42,model='diffusion',**kw);b=build_model(42,model='diffusion',**kw)
    b.load_state_dict(a.state_dict())
    x=torch.randint(0,2,(8,1,8,8,8)).float()*2-1
    a.forward_a(x).sum().backward()
    assert a.cond_feat.weight.grad is not None
    assert torch.isfinite(a.cond_feat.weight.grad).all()
    assert a.cond_feat.weight.grad.abs().sum()>0
