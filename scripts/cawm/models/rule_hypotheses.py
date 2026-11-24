"""Prefix-only, persistent-rule multi-hypothesis decoder.

Uniform independent prior over unobserved rule classes, NOT the actual
held-out half or the self-consistency filter. Each new class bit is drawn
once for an entire trajectory. No future observations enter decoding.
Exact best-first MAP under this prior if a leaf is reached within budget.
"""
import heapq
import itertools
import numpy as np
from cawm.simulate import neighbor_count

SHIFTS=tuple((a,b) for a in (-1,0,1) for b in (-1,0,1))
def raw_pattern(x):
    x=np.asarray(x,dtype=np.int16)
    return sum(np.roll(x,(dy,dx),axis=(-2,-1))*(1<<i)
               for i,(dy,dx) in enumerate(SHIFTS))

def outer_totalistic_classes():
    # Diagnostic structured reference; not supplied to learned partition.
    bits=((np.arange(512)[:,None]>>np.arange(9))&1)
    return (9*bits[:,4]+bits.sum(1)-bits[:,4]).astype(np.int16)

def learned_classes(kernel,threshold=.5):
    """Freeze connected components of learned affinity graph, no target labels."""
    kernel=np.asarray(kernel)
    if kernel.shape!=(512,512) or not np.isfinite(kernel).all() or not 0<threshold<1:
        raise ValueError("invalid kernel")
    if not np.allclose(kernel,kernel.T,atol=1e-5):
        raise ValueError("asymmetric affinity")
    groups=np.full(512,-1,dtype=np.int16)
    n=0
    for i in range(512):
        if groups[i]>=0:continue
        groups[i]=n; stack=[i]
        while stack:
            j=stack.pop()
            pending=np.flatnonzero((kernel[j]>=threshold)&(groups<0))
            groups[pending]=n;stack.extend(pending.tolist())
        n+=1
    return groups

def prefix_table(prefix,classes):
    prefix=np.asarray(prefix)
    classes=np.asarray(classes)
    if prefix.ndim!=3 or prefix.shape[0]!=8 or min(prefix.shape[1:])<3 or not np.isin(prefix,[0,1]).all():
        raise ValueError("expected 8 observed binary frames")
    if classes.shape!=(512,) or classes.min()<0 or not np.issubdtype(classes.dtype,np.integer):
        raise ValueError("invalid class map")
    q=classes[raw_pattern(prefix[:7])].ravel()
    y=prefix[1:].ravel().astype(np.int16)
    t=np.full(int(classes.max())+1,-1,dtype=np.int16)
    t[q]=y
    if not np.array_equal(t[q],y):
        return None  # cannot honestly decode a contradictory learned partition
    return t

def marginal_first(prefix,classes):
    t=prefix_table(prefix,classes)
    if t is None:return None
    p=t[classes[raw_pattern(prefix[-1])]]
    return np.where(p<0,.5,p)

def decode(prefix,classes,steps=8,top_k=1,max_nodes=4096):
    """Return highest-probability unique futures; budget exhaustion explicit.

    Search nodes are partial assignments to previously unseen class bits.
    A path with k such assignments has probability 2**(-k). Deterministic
    frame propagation stops at the first required unknown class. Equal-prior
    branches are expanded in fixed order. Heap priority is monotone, so any
    emitted complete path is globally MAP (or next-ranked) even if later
    search hits budget. Remaining posterior mass is not silently discarded.
    """
    if min(steps,top_k,max_nodes)<1:raise ValueError("positive budgets")
    classes=np.asarray(classes)
    t=prefix_table(prefix,classes)
    if t is None:return dict(paths=[],nodes=0,exhausted=False,conflict=True)
    heap=[];tie=itertools.count()
    # Heap state holds already generated frames, last frame, and table.
    heapq.heappush(heap,(0,next(tie),t,np.asarray(prefix[-1],dtype=np.uint8),()))
    paths=[];nodes=0
    while heap and len(paths)<top_k and nodes<max_nodes:
        k,_,t,cur,frames=heapq.heappop(heap);nodes+=1
        while len(frames)<steps:
            q=classes[raw_pattern(cur)]
            unknown=np.unique(q[t[q]<0])
            if unknown.size:
                j=int(unknown[0])
                for v in (0,1):
                    tt=t.copy();tt[j]=v
                    heapq.heappush(heap,(k+1,next(tie),tt,cur,frames))
                break
            cur=t[q].astype(np.uint8)
            frames=frames+(cur,)
        else:
            paths.append(dict(frames=np.stack(frames),new_bits=k,probability=2.**(-k)))
    return dict(paths=paths,nodes=nodes,conflict=False,
                exhausted=bool(heap) and len(paths)<top_k)

def merge_observed_classes(classes,prefixes,min_overlap=12):
    """Merge learned subgroups only with repeated, conflict-free evidence.

    Uses exclusively training histories. Pairs must co-occur in >=min_overlap
    independent trajectories and always agree there. Recompute the constraints
    after each merge, so a transitive merge cannot introduce a known conflict.
    This is a heuristic, not proof of rule equivalence under SC selection.
    """
    classes=np.asarray(classes).copy()
    prefixes=np.asarray(prefixes)
    if prefixes.ndim!=4 or prefixes.shape[1]!=8 or min_overlap<1:
        raise ValueError("training histories only; positive overlap")
    tabs=[]
    for x in prefixes:
        t=prefix_table(x,classes)
        if t is None:raise ValueError("contradictory input partition")
        tabs.append(t)
    t=np.stack(tabs)
    active=list(range(t.shape[1]));merges=[]
    while True:
        best=None
        for i,a in enumerate(active):
            for b in active[i+1:]:
                joint=(t[:,a]>=0)&(t[:,b]>=0)
                n=int(joint.sum())
                if n<min_overlap or (t[joint,a]!=t[joint,b]).any():continue
                if best is None or n>best[0]:best=(n,a,b)
        if best is None:break
        n,a,b=best
        t[:,a]=np.where(t[:,a]>=0,t[:,a],t[:,b])
        classes[classes==b]=a;active.remove(b)
        merges.append((a,b,n))
    _,out=np.unique(classes,return_inverse=True)
    return out.astype(np.int16),merges
