import itertools
import numpy as np
import pytest
from cawm.models import mutating_rule as m, noisy_posterior as s
from cawm.models.rule_hypotheses import raw_pattern

C=np.arange(512)%3
def sample():
    return np.random.default_rng(3).integers(0,2,(5,10,4,4),dtype=np.uint8)

def test_static_reduction():
    x=sample()
    np.testing.assert_allclose(m.sequence_scores(x,C,.17,0),s.sequence_scores(x,C,.17)[0],atol=1e-10)

def test_exact_enumeration():
    # One rule bit, same value shared across all pixels in a transition.
    x=sample()[:1,:9]; p=.13;q=.21
    y=x[0,1:].reshape(8,-1); n=y.sum(1);d=y.shape[1]
    def likelihood(T):
        total=0.
        for bits in itertools.product((0,1),repeat=T):
            prob=.5
            for t,b in enumerate(bits):
                if t:prob*=q if b!=bits[t-1] else 1-q
                wrong=d-n[t] if b else n[t]
                prob*=p**wrong*(1-p)**(d-wrong)
            total+=prob
        return total
    expected=-np.log(likelihood(8)/likelihood(7))
    np.testing.assert_allclose(m.sequence_scores(x,np.zeros(512,int),p,q)[0,0],expected,atol=1e-10)

def test_output_coin_vs_rule_coin():
    x=np.zeros((2,10,4,4),np.uint8);c=np.zeros(512,int)
    # Fair rule bit changes cost one bit per frame, not 16 bits.
    np.testing.assert_allclose(m.sequence_scores(x,c,0,.5),np.log(2))
    np.testing.assert_allclose(m.sequence_scores(x,c,.5,0),16*np.log(2))
    x[:,8,0,0]=1
    a=m.sequence_scores(x,c,0,.5)
    assert np.isposinf(a[:,0]).all() and np.isnan(a[:,1]).all()

def test_prefix_and_rollout():
    x=sample();g=[(.1,0),(.2,.1),(.5,.5)]
    r,score=m.fit_rates(x[:,:8],C,g)
    y=x.copy();y[:,8:]^=1
    np.testing.assert_array_equal(m.posterior(x[:,:8],C,*r),m.posterior(y[:,:8],C,*r))
    np.testing.assert_array_equal(m.rollout(x[:,:8],C,*r,7),m.rollout(y[:,:8],C,*r,7))
    with pytest.raises(ValueError):m.fit_rates(x,C,g)
    with pytest.raises(ValueError):m.posterior(x[:,:8],C,-1,0)
    with pytest.raises(ValueError):m.posterior(x[:,:8],C,0,0)

def test_torus_corner_negative_control():
    x=np.zeros((5,16,4,4),np.uint8);x[:,0]=np.random.default_rng(5).integers(0,2,(5,4,4))
    for t in range(15):x[:,t+1]=np.roll(x[:,t],(1,1),axis=(-2,-1))
    z=np.zeros((1,4,4),np.uint8);z[0,-1,-1]=1
    bit=raw_pattern(z)[0,0,0];good=((np.arange(512)&bit)!=0).astype(int)
    center=np.zeros_like(z);center[0,0,0]=1
    bad=((np.arange(512)&raw_pattern(center)[0,0,0])!=0).astype(int)
    np.testing.assert_array_equal(m.rollout(x[:,:8],good,0,0,3),x[:,8:])
    assert m.sequence_scores(x,good,.01,.01).sum()<m.sequence_scores(x,bad,.1,.1).sum()


def test_rule_rate_unidentifiable_with_fair_output_noise():
    x=sample()
    _,scores=m.fit_rates(x[:,:8],C,[(.5,0),(.5,.1),(.5,.5)])
    np.testing.assert_allclose(scores,scores[0],atol=1e-10)
    np.testing.assert_allclose(m.sequence_scores(x,C,.5,.5),16*np.log(2))

def test_invalid_rates_and_no_future_leak_in_first_likelihood():
    x=sample();y=x.copy();y[:,9:]^=1
    np.testing.assert_array_equal(m.sequence_scores(x,C,.1,.2)[:,0],m.sequence_scores(y,C,.1,.2)[:,0])
    with pytest.raises(ValueError):m.fit_rates(x[:,:8],C,[(.1,float("nan"))])
