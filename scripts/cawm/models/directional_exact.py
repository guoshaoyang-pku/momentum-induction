"""Bounded branch-and-bound MAP over prefix-compatible persistent CA rules.

The fair independent prior is the same as directional_search. We branch only
on newly queried rule slots, never on unused completions. Each completed
path has mass 2**(-cost); costs are nondecreasing. A certificate means no
path with a smaller cost exists, not that the hidden rule is identifiable.
"""
import numpy as np
from .directional_metric import OFFSETS
from .directional_search import canonical_classes, prefix_tables, search

def exact_search(prefix, classes, *, seed=0, candidates=32, pool=128, max_nodes=4096, horizon=8):
    if not isinstance(max_nodes,int) or max_nodes<0:
        raise ValueError("nonnegative integer node budget required")
    if not 1<=candidates<=pool:raise ValueError("invalid candidate pool")
    classes=canonical_classes(classes)
    tables=prefix_tables(prefix,classes)
    incumbent,costs,_=search(prefix,classes,tuple(sorted({candidates,pool})),seed=seed,horizon=horizon)[candidates]
    baseline=costs.copy()
    incumbent=incumbent.copy()
    certified=np.zeros(len(tables),bool)
    visited=np.zeros(len(tables),int)
    h,w=prefix.shape[-2:]
    ids=np.arange(h*w).reshape(h,w)
    indices=np.stack([np.roll(ids,(dy,dx),axis=(0,1)).ravel() for dy,dx in OFFSETS])
    powers=(1<<np.arange(9))[:,None]
    def query(cur):
        return classes[(cur.ravel()[indices]*powers).sum(0)].astype(np.uint64)
    def mask(q):
        return int(np.bitwise_or.reduce(np.left_shift(np.uint64(1),q)))
    for i,t in enumerate(tables):
        known=sum(1<<int(j) for j in np.flatnonzero(t>=0))
        ones=sum(1<<int(j) for j in np.flatnonzero(t==1))
        cur=prefix[i,-1]
        lower=(mask(query(cur))&~known).bit_count()
        best=int(costs[i]); nodes=0; exhausted=False
        if best<lower: raise RuntimeError("invalid incumbent")
        def dfs(cur,known,ones,cost,path):
            nonlocal best,nodes,exhausted
            if best==lower or cost>=best or exhausted:return
            if nodes>=max_nodes:
                exhausted=True;return
            nodes+=1
            if len(path)==horizon:
                best=cost;incumbent[i]=np.stack(path)
                return
            q=query(cur)
            need=mask(q)&~known
            new_cost=cost+need.bit_count()
            if new_cost>=best:return
            subset=0
            while True:
                updated=ones|subset
                nxt=((np.uint64(updated)>>q)&1).astype(np.uint8).reshape(h,w)
                dfs(nxt,known|need,updated,new_cost,path+[nxt])
                if exhausted or best==lower or subset==need:break
                subset=(subset-need)&need
        dfs(cur,known,ones,0,[])
        costs[i]=best
        visited[i]=nodes
        certified[i]=not exhausted or best==lower
    return incumbent,costs,certified,visited,baseline
