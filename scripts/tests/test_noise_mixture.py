import numpy as np
import pytest
from cawm.models import noise_mixture as m, noisy_posterior as single
from cawm.models.rule_hypotheses import raw_pattern

C=np.arange(512)%18
def sample():
    return np.random.default_rng(7).integers(0,2,(4,16,4,4),dtype=np.uint8)

def test_single_component():
    x=sample()
    np.testing.assert_allclose(m.sequence_scores(x,C,[.2],[1]),single.sequence_scores(x,C,.2)[0],atol=1e-10)
def test_exact_joint_mixture():
    x=sample();g=np.array([.1,.3]);w=np.array([.2,.8])
    post,_=m.posterior(x[:,:8],C,g,w)
    losses=np.stack([single.sequence_scores(x,C,p)[0] for p in g],1)
    expected=-np.log((post*np.exp(-losses.sum(2))).sum(1))
    np.testing.assert_allclose(m.sequence_scores(x,C,g,w).sum(1),expected,atol=1e-10)
def test_fair_coin():
    x=sample()
    np.testing.assert_allclose(m.sequence_scores(x,C,[.5],[1]),16*np.log(2))
def test_prefix_and_em():
    x=sample();g=[.1,.3,.5]
    w,t=m.fit_prior(x[:,:8],C,g)
    assert np.all(np.diff(t)>=-1e-7) and abs(w.sum()-1)<1e-8
    altered=x.copy();altered[:,8:]^=1
    np.testing.assert_array_equal(m.posterior(x[:,:8],C,g,w)[0],m.posterior(altered[:,:8],C,g,w)[0])
    with pytest.raises(ValueError):m.fit_prior(x,C,g)
def test_rollout_reproducible_and_torus():
    x=sample();g=[.1,.3,.5];w=[.2,.3,.5]
    a,p=m.rollout(x[:,:8],C,g,w,5)
    b,q=m.rollout(x[:,:8],C,g,w,5)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,q)
    assert a.shape==(4,8,4,4) and set(p)<=set(g)
    z=np.zeros((1,4,4),dtype=np.uint8);z[0,-1,-1]=1
    assert raw_pattern(z)[0,0,0]!=0
def test_invalid_prior():
    with pytest.raises(ValueError):m.posterior(sample()[:,:8],C,[.1,.3],[1,1])

def test_deterministic_rate_persists_across_rollout():
    # All-zero history and one class: the p=0 posterior fixes its rule to zero.
    x=np.zeros((16,8,4,4),dtype=np.uint8);classes=np.zeros(512,dtype=int)
    out,p=m.rollout(x,classes,[0,.5],[.5,.5],123,steps=32)
    assert np.all(p==0) and not out.any()
def test_wrong_spatial_class_is_negative_control():
    # A wrap-around shift rule has a consistent table when indexed by that
    # corner, but the center-only model cannot explain the same history.
    rng=np.random.default_rng(8)
    x=np.zeros((8,16,4,4),dtype=np.uint8);x[:,0]=rng.integers(0,2,(8,4,4))
    for t in range(15):x[:,t+1]=np.roll(x[:,t],(1,1),axis=(-2,-1))
    # raw_pattern uses fixed bit order; derive the corner slot using a probe.
    z=np.zeros((1,4,4),dtype=np.uint8);z[0,-1,-1]=1
    bit=int(raw_pattern(z)[0,0,0]);assert bit>0 and bit&(bit-1)==0
    good=(np.arange(512)&bit)!=0
    center=np.zeros_like(z);center[0,0,0]=1
    bad=(np.arange(512)&int(raw_pattern(center)[0,0,0]))!=0
    grid=[0,.5];weights=[.5,.5]
    a=m.sequence_scores(x,good.astype(int),grid,weights).sum()
    b=m.sequence_scores(x,bad.astype(int),grid,weights).sum()
    assert a<1e-8 and b>100
