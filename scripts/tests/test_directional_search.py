import itertools
import numpy as np
import pytest
from cawm.models.directional_search import search,prefix_tables,metric_classes
from cawm.models.directional_metric import patterns
from run_directional_metric import dataset,SLOTS

def test_partition_learned():
    # Independent weights with separated groups; decoder never receives SLOTS.
    w=np.zeros((3,9)); w[0,4]=2; w[1,[1,3,5,7]]=2; w[2,[0,2,6,8]]=2
    c=metric_classes(w)
    assert np.array_equal(c[:,None]==c[None,:],SLOTS[:,None]==SLOTS[None,:])

def test_prefix_only_and_persistent():
    x,_=dataset(42,12)
    a=search(x[:,:8],SLOTS,(1,8),seed=7)
    changed=x.copy();changed[:,8:]^=1
    b=search(changed[:,:8],SLOTS,(1,8),seed=7)
    t=prefix_tables(x[:,:8],SLOTS)
    for k,(p,c,cert) in a.items():
        assert all(np.array_equal(u,v) for u,v in zip(a[k],b[k]))
        for i in range(len(x)):
            table=t[i].copy(); prev=x[i,7]; unknown=0
            for frame in p[i]:
                q=SLOTS[patterns(prev)]
                for j,v in zip(q.ravel(),frame.ravel()):
                    if table[j]<0: table[j]=v;unknown+=1
                    assert table[j]==v
                prev=frame
            assert unknown==c[i]
    assert np.all(a[8][1]<=a[1][1])
    with pytest.raises(ValueError):search(x,SLOTS,(1,))

def test_small_exact_certificate():
    # A one-class CA with a constant observed rule: no new bits and mass one.
    x=np.ones((2,8,8,8),np.uint8);c=np.zeros(512,int)
    for p,cost,cert in search(x,c,(1,8)).values():
        assert p.all() and not cost.any() and cert.all()

def test_corner_index():
    x=np.zeros((8,8),np.uint8);x[7,7]=1
    assert patterns(x)[0,0]==256

def test_class_relabeling_and_conflict():
    x,_=dataset(55,9)
    a=search(x[:,:8],SLOTS,(1,8),seed=10)
    b=search(x[:,:8],49-SLOTS,(1,8),seed=10)
    assert all(np.array_equal(u,v) for k in a for u,v in zip(a[k],b[k]))
    with pytest.raises(ValueError):prefix_tables(x[:,:8],np.zeros(512,int))

def test_exact_small_rule_enumeration():
    # Independent brute enumeration over two local bits. Verify cost and
    # the claimed certificates against all possible rule completions.
    c=(np.arange(512)>>4&1)*2+(np.arange(512)>>1&1)
    rng=np.random.default_rng(54)
    for _ in range(24):
        rule=rng.integers(0,2,4,dtype=np.uint8)
        p=[rng.integers(0,2,(3,3),dtype=np.uint8)]
        for j in range(7): p.append(rule[c[patterns(p[-1])]])
        p=np.array(p)[None]
        t=prefix_tables(p,c)[0]
        unknown=np.flatnonzero(t<0)
        paths={}
        for bits in itertools.product((0,1),repeat=len(unknown)):
            a=t.copy();a[unknown]=bits
            cur=p[0,-1]; frames=[]
            for j in range(4):
                cur=a[c[patterns(cur)]].astype(np.uint8);frames.append(cur)
            key=np.asarray(frames).tobytes()
            paths[key]=paths.get(key,0)+1
        for pred,cost,cert in search(p,c,(1,8),seed=2,horizon=4).values():
            mass=paths[pred[0].tobytes()]/(2**len(unknown))
            assert mass==2.**(-int(cost[0]))
            if cert[0]:assert paths[pred[0].tobytes()]==max(paths.values())
