import numpy as np
import pytest
from cawm.models.rule_hypotheses import decode,outer_totalistic_classes,prefix_table
from cawm.simulate import simulate
from run_rule_prior import estimate,score_first

def test_weighted_search_exact():
    rng=np.random.default_rng(90);c=outer_totalistic_classes()
    prior=np.linspace(.12,.88,18)
    for _ in range(12):
        x=simulate([int(rng.integers(1<<18))],rng.integers(0,2,(1,4,4),dtype=np.uint8),7)[0]
        t=prefix_table(x,c);missing=np.flatnonzero(t<0)
        if not 0<len(missing)<=10:continue
        ids=np.arange(1<<len(missing));tab=np.tile(t,(len(ids),1))
        tab[:,missing]=(ids[:,None]>>np.arange(len(missing)))&1
        weights=np.where(tab[:,missing],prior[missing],1-prior[missing]).prod(1)
        y=simulate(tab@(1<<np.arange(18)),np.repeat(x[-1:],len(ids),axis=0),2)[:,1:]
        _,inv=np.unique(y.reshape(len(y),-1),axis=0,return_inverse=True)
        expected=np.sort(np.bincount(inv,weights=weights))[::-1][:4]
        got=decode(x,c,2,4,4096,prior=prior)
        np.testing.assert_allclose([v["probability"] for v in got["paths"]],expected)
        return
    pytest.fail("no uncertain prefix")

def test_one_vote_and_leakage():
    c=outer_totalistic_classes();x=np.zeros((3,8,4,4),np.uint8)
    p,n=estimate(x,c)
    assert n[0]==3 and p[0]==.2
    assert (n[1:]==0).all() and (p[1:]==.5).all()
    with pytest.raises(ValueError):estimate(np.concatenate([x,x],axis=1),c)
    with pytest.raises(ValueError):decode(x[0],c,prior=np.zeros(18))

def test_joint_vs_marginal_and_impossible():
    c=outer_totalistic_classes();rng=np.random.default_rng(7);p=np.full(18,.5)
    from cawm.models.rule_hypotheses import raw_pattern
    for _ in range(100):
        seq=simulate([int(rng.integers(1<<18))],rng.integers(0,2,(1,8,8),dtype=np.uint8),8)[0]
        q=c[raw_pattern(seq[7])];t=prefix_table(seq[:8],c)
        unknown=q[t[q]<0]
        if len(unknown)<=len(np.unique(unknown)):continue
        joint,marg=score_first(seq[:8],seq[8],c,p)
        assert joint<marg
        bad=seq[8].copy();j=unknown[0];inds=np.flatnonzero(q.ravel()==j)
        if len(inds)<2:continue
        bad.ravel()[inds[0]]^=1
        assert score_first(seq[:8],bad,c,p)[0] is None
        return
    pytest.fail("no repeated unknown query")
