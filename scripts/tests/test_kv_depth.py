import torch
from cawm.models.art_kvshift import ArtKVShift
from cawm.models.art_induction import ArtInduction
from cawm.train import build_model, model_kwargs


def test_single_read_default_identity_and_parameter_control():
    torch.manual_seed(42)
    m=ArtKVShift().eval()
    torch.manual_seed(42)
    n=ArtKVShift(reads=2).eval()
    assert m.total_param_count()==2625
    assert n.total_param_count()==ArtKVShift(head_hidden=87).total_param_count()==5019
    for key,val in m.state_dict().items():
        if not key.startswith('head.'):
            assert torch.equal(val,n.state_dict()[key])
    q=torch.randn(2,16,18);tok=torch.randn(2,112,36)
    with torch.no_grad():
        assert torch.equal(m._logits_from(tok,q),ArtInduction._logits_from(m,tok,q))


def test_second_read_gradient_causality_and_reload():
    a=dict(seed=43,model='art_kvshift',grid=4,arm='emergence',head='concat',
           kv_reads=2,kv_head_hidden=24)
    m=build_model(43,model='art_kvshift',**model_kwargs(a))
    x=torch.randint(0,2,(2,1,16,4,4)).float()*2-1
    z=m(x);assert z.shape==(2,8,4,4)
    z.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    n=build_model(43,model='art_kvshift',**model_kwargs(a));n.load_state_dict(m.state_dict());n.eval()
    changed=x.clone();changed[:,:,8:]*=-1
    with torch.no_grad():
        assert torch.equal(m(x)[:,0],m(changed)[:,0])
        assert torch.equal(m.rollout(x[:,:,:8]),n.rollout(x[:,:,:8]))
        p=m.rollout(x[:,:,:8]);assert p.shape==(2,8,4,4)
        assert torch.equal(p[:,0],(m(x)[:,0]>0).to(torch.uint8))
