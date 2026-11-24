import numpy as np
import pytest
from cawm.models.noisy_posterior import counts,posterior,sequence_scores,fit_noise,rollout
from cawm.models.rule_hypotheses import outer_totalistic_classes,raw_pattern

def test_corner_wrap_and_counts():
    x=np.zeros((1,8,4,4),np.uint8);x[0,0,3,3]=1
    c=outer_totalistic_classes()
    assert c[raw_pattern(x[0,0])[0,0]]==1
    z,o=counts(x,c);assert z.sum()+o.sum()==7*16

def test_joint_enumeration_repeated_queries():
    c=np.zeros(512,dtype=np.int16);p=.13
    x=np.zeros((1,10,3,3),np.uint8);x[:,1:8].reshape(-1)[:31]=1; x[:,8:]=1
    def prob(a):
        y=a[:,1:];n=y.size;s=y.sum()
        return .5*((1-p)**(n-s)*p**s+p**(n-s)*(1-p)**s)
    j,m,_=sequence_scores(x,c,p)
    np.testing.assert_allclose(j.sum(),-np.log(prob(x)/prob(x[:,:8])),rtol=1e-12)
    assert not np.isclose(j[0,0],m[0,0])

def test_fair_coin():
    rng=np.random.default_rng(42);x=rng.integers(0,2,(3,16,4,4),dtype=np.uint8)
    c=outer_totalistic_classes();j,m,_=sequence_scores(x,c,.5)
    np.testing.assert_allclose(j,16*np.log(2));np.testing.assert_allclose(m,j)
    np.testing.assert_allclose(posterior(x[:,:8],c,.5),.5)

def test_deterministic_and_future_guard():
    c=outer_totalistic_classes();x=np.zeros((2,16,4,4),np.uint8)
    assert fit_noise(x[:,:8],c)==0
    j,m,_=sequence_scores(x,c,0);assert (j==0).all() and (m==0).all()
    bad=x.copy();bad[:,9,0,0]=1
    jb,_,_=sequence_scores(bad,c,0);assert np.isinf(jb[:,1]).all() and np.isnan(jb[:,2:]).all()
    np.testing.assert_array_equal(rollout(x[:,:8],c,.1,42,True),rollout(bad[:,:8],c,.1,42,True))
    with pytest.raises(ValueError):rollout(x,c,.1)
    with pytest.raises(ValueError):fit_noise(x,c)

def test_fit_noise_and_negative_control():
    rng=np.random.default_rng(9);x=np.zeros((128,8,4,4),np.uint8)
    x[:,1:]=rng.random((128,7,4,4))<.1
    c=np.zeros(512,dtype=np.int16)
    assert abs(fit_noise(x,c)-.1)<.01
    a=posterior(x,c,.1);assert a.mean()<.01
    b=posterior(x,c,.5);np.testing.assert_allclose(b,.5)
