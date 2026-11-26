import itertools
import numpy as np
from cawm.models.directional_exact import exact_search
from cawm.models.directional_search import prefix_tables
from cawm.models.directional_metric import patterns
from run_directional_metric import dataset,SLOTS

def test_exact_matches_independent_enumeration():
    rng=np.random.default_rng(91)
    c=(np.arange(512)>>4&1)*4+(np.arange(512)>>1&1)*2+(np.arange(512)>>7&1)
    for _ in range(48):
        rule=rng.integers(0,2,8,dtype=np.uint8)
        x=[rng.integers(0,2,(3,3),dtype=np.uint8)]
        for j in range(7):x.append(rule[c[patterns(x[-1])]])
        p=np.array(x)[None]
        table=prefix_tables(p,c)[0]
        missing=np.flatnonzero(table<0)
        mass={}
        for bits in itertools.product((0,1),repeat=len(missing)):
            a=table.copy();a[missing]=bits
            cur=p[0,-1];out=[]
            for j in range(4):
                cur=a[c[patterns(cur)]].astype(np.uint8);out.append(cur)
            key=np.asarray(out).tobytes()
            mass[key]=mass.get(key,0)+1
        pred,cost,cert,nodes,base=exact_search(p,c,candidates=1,horizon=4,max_nodes=4096)
        assert cert[0] and nodes[0]<=4096 and cost[0]<=base[0]
        assert mass[pred[0].tobytes()]==max(mass.values())
        assert mass[pred[0].tobytes()]/2**len(missing)==2.**(-int(cost[0]))
        # A capped run must not claim a certificate for a suboptimal path.
        pred,cost,cert,nodes,base=exact_search(p,c,candidates=1,horizon=4,max_nodes=0)
        assert nodes[0]==0
        if cert[0]:assert mass[pred[0].tobytes()]==max(mass.values())

def test_prefix_invariance_and_relabel():
    x,_=dataset(123,24)
    a=exact_search(x[:,:8],SLOTS,max_nodes=128,candidates=4)
    x[:,8:]^=1
    b=exact_search(x[:,:8],49-SLOTS,max_nodes=128,candidates=4)
    assert all(np.array_equal(u,v) for u,v in zip(a,b))
    assert np.all(a[1]<=a[4]) and np.max(a[3])<=128
