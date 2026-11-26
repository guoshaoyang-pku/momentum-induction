import itertools
import numpy as np
import pytest
from cawm.models.directional_metric import patterns
from cawm.models.directional_search import prefix_tables, search
from run_directional_history import likelihood, evaluate
from run_directional_metric import dataset,SLOTS

def test_fixed_target_more_observations():
    x,_=dataset(101,24)
    counts=[];costs=[]
    for h in (1,2,4,8):
        p=x[:,8-h:8]; f=x[:,8:]
        t=prefix_tables(p,SLOTS,history=h)
        counts.append((t>=0).sum(1))
        b,u=likelihood(p,f,SLOTS)
        costs.append(b.sum(1))
        assert np.all(b<=u)
        a=search(p,SLOTS,(2,),seed=8,history=h)[2]
        x2=x.copy();x2[:,8:]^=1
        z=search(x2[:,8-h:8],SLOTS,(2,),seed=8,history=h)[2]
        assert all(np.array_equal(q,r) for q,r in zip(a,z))
    assert np.all(counts[0]==0)
    assert np.all(np.diff(counts,axis=0)>=0)
    assert np.all(np.diff(costs,axis=0)<=0)
    with pytest.raises(ValueError):prefix_tables(x[:,:4],SLOTS)
    with pytest.raises(ValueError):search(x[:,:9],SLOTS)

def test_likelihood_matches_exhaustive_rule_completions():
    c=((np.arange(512)>>4)&1)*2+((np.arange(512)>>1)&1)
    rng=np.random.default_rng(12)
    for _ in range(16):
        rule=rng.integers(0,2,4,dtype=np.uint8)
        x=[rng.integers(0,2,(3,3),dtype=np.uint8)]
        for _ in range(5):x.append(rule[c[patterns(x[-1])]])
        x=np.array(x)
        for h in (1,2,3):
            p=x[3-h:3][None];f=x[3:][None]
            t=prefix_tables(p,c,history=h)[0]
            missing=np.flatnonzero(t<0); hits=0
            for vals in itertools.product((0,1),repeat=len(missing)):
                r=t.copy();r[missing]=vals
                cur=p[0,-1];out=[]
                for _ in range(3):
                    cur=r[c[patterns(cur)]].astype(np.uint8);out.append(cur)
                hits+=np.array_equal(out,f[0])
            bits,_=likelihood(p,f,c)
            assert hits/2**len(missing)==2.**(-int(bits.sum()))
