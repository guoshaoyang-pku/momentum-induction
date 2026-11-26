import math
import pytest
import torch
from cawm.models import art_vanilla as art, arcnn
from cawm.models.torus_mixture import (
    component_log_probs, mixture_log_prob, shifts_for,
    sample_next_frame, rollout_sample, _frame_logits,
)

@pytest.fixture
def models(monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setitem(art.LAYERS,8,1)
    monkeypatch.setattr(arcnn,"CH",4)
    return [art.ArtVanilla(grid=4,dmodel=8,nlayers=1,nhead=4).eval(),
            arcnn.ARCNN(grid=4).eval()]

def data():
    gen=torch.Generator().manual_seed(19)
    return 2*torch.randint(2,(2,8,4,4),generator=gen).float()-1,torch.randint(2,(2,4,4),generator=gen)

def test_subgroup_invariance_and_cnn_equivariance(models):
    x,y=data()
    for m in models:
        base=component_log_probs(m,x,y,2)
        assert base.shape==(2,4)
        mix=mixture_log_prob(m,x,y,2)
        torch.testing.assert_close(mix,torch.logsumexp(base,1)-math.log(4))
        assert bool((mix>=base.mean(1)-1e-8).all())
        for shift in shifts_for(4,2):
            torch.testing.assert_close(mix,mixture_log_prob(m,x.roll(shift,(-2,-1)),y.roll(shift,(-2,-1)),2),atol=1e-6,rtol=1e-6)
    cnn=component_log_probs(models[1],x,y,1)
    torch.testing.assert_close(cnn,cnn[:,0:1].expand_as(cnn),atol=1e-5,rtol=1e-6)

def test_raw_joint_chain_and_no_target_leak(models):
    x,y=data()
    for m in models:
        z=_frame_logits(m,x,y)
        y2=y.clone(); y2[:, -1,-1]^=1
        torch.testing.assert_close(z,_frame_logits(m,x,y2))
        if isinstance(m,art.ArtVanilla):
            y2=y.clone();y2[:,0,0]^=1
            torch.testing.assert_close(z[:,0,0],_frame_logits(m,x,y2)[:,0,0])
        else:
            torch.testing.assert_close(z,_frame_logits(m,x,1-y))
        frames=torch.cat([x,(2*y.float()-1)[:,None],torch.ones(2,7,4,4)],1)
        torch.testing.assert_close(z,m(frames[:,None])[:,0],atol=2e-6,rtol=2e-5)

def test_corner_roll_and_sampling(models):
    x=torch.full((2,8,4,4),-1.)
    x[:,:,-1,-1]=1
    assert (x.roll((1,1),(-2,-1))[:,:,0,0]==1).all()
    assert shifts_for(4,4)==((0,0),)
    for m in models:
        a=sample_next_frame(m,x,2,torch.Generator().manual_seed(3))
        b=rollout_sample(m,x,2,2,torch.Generator().manual_seed(3))
        assert a.dtype==torch.uint8 and b.shape==(2,2,4,4)
        assert torch.equal(a,b[:,0])
        assert bool(((b==0)|(b==1)).all())
    with pytest.raises(ValueError): shifts_for(4,3)
    with pytest.raises(ValueError): mixture_log_prob(models[0],x,torch.full((2,4,4),2),2)


def test_full_group_raw_noninvariance(models):
    # A raw learned-position transformer is not structurally equivariant.
    # The joint-probability mixture is, including one-cell seam crossings.
    x,y=data()
    m=models[0]
    z=component_log_probs(m,x,y,1)
    assert z.shape==(2,16)
    assert float((z.max(1).values-z.min(1).values).max())>1e-4
    q=mixture_log_prob(m,x,y,1)
    for s in [(1,0),(0,1),(1,1),(-1,-1)]:
        torch.testing.assert_close(q,mixture_log_prob(m,x.roll(s,(-2,-1)),y.roll(s,(-2,-1)),1),atol=1e-6,rtol=1e-6)


def test_actual_diagonal_seam_and_wrong_center(models):
    # Program a simple valid CA into the existing R0 CNN. The NW neighbor
    # of (0,0) is (-1,-1), so this rule translates by (+1,+1).
    m=models[1]
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.c1.weight[0,7,0,0]=1
        for conv in [m.c2,m.c3,m.c4]:
            conv.weight[0,0,1,1]=1
        m.out.weight[0,0,0,0]=200
        m.out.bias[0]=-100
    x=torch.full((2,8,4,4),-1.)
    x[:,:,-1,-1]=1
    truth=torch.stack([((x[:,-1].roll((k,k),(-2,-1))+1)/2).to(torch.uint8) for k in (1,2,3)],1)
    actual=rollout_sample(m,x,3,1,torch.Generator().manual_seed(7))
    assert actual[:,0,0,0].all() and torch.equal(actual,truth)
    # Deliberately wrong center-copy rule; same shapes, same padding.
    with torch.no_grad():
        m.c1.weight.zero_()
        m.c1.weight[0,7,1,1]=1
    wrong=rollout_sample(m,x,3,1,torch.Generator().manual_seed(7))
    assert not torch.equal(wrong,truth)
    assert not wrong[:,0,0,0].any()
    assert (mixture_log_prob(m,x,truth[:,0],1)<-100).all()
