"""Prefix-only persistent-rule search; no target data enters the decoder."""
import numpy as np
from .directional_metric import patterns

def metric_classes(weight, threshold=.5):
    raw=2*((np.arange(512)[:,None]>>np.arange(9))&1)-1
    e=np.einsum('ni,di->nd',raw,np.asarray(weight,dtype=float))
    k=np.exp(-((e[:,None]-e[None,:])**2).sum(-1))
    classes=np.full(512,-1,dtype=int)
    group=0
    for i in range(512):
        if classes[i]>=0: continue
        classes[i]=group
        stack=[i]
        while stack:
            j=stack.pop()
            ix=np.flatnonzero((k[j]>=threshold)&(classes<0))
            classes[ix]=group
            stack.extend(ix.tolist())
        group+=1
    return classes

def canonical_classes(classes):
    """Relabel in first raw-pattern order, independent of original labels."""
    classes=np.asarray(classes)
    if classes.shape!=(512,) or not np.issubdtype(classes.dtype,np.integer) or classes.min()<0:
        raise ValueError('invalid class map')
    _,first,inv=np.unique(classes,return_index=True,return_inverse=True)
    return np.argsort(np.argsort(first))[inv]

def prefix_tables(prefix, classes, *, history=8):
    prefix=np.asarray(prefix)
    classes=np.asarray(classes)
    if type(history) is not int or history<1 or prefix.ndim!=4 or len(prefix)==0 or prefix.shape[1]!=history or not np.isin(prefix,[0,1]).all():
        raise ValueError('explicit number of binary prefix frames required')
    if classes.shape!=(512,) or not np.issubdtype(classes.dtype,np.integer) or not np.array_equal(np.unique(classes),np.arange(classes.max()+1)) or classes.max()>=64:
        raise ValueError('need contiguous classes, at most 64')
    q=classes[patterns(prefix[:,:-1])].reshape(len(prefix),-1)
    y=prefix[:,1:].reshape(len(prefix),-1)
    t=np.full((len(prefix),int(classes.max())+1),-1,dtype=np.int8)
    i=np.arange(len(prefix))[:,None]
    t[i,q]=y
    if not np.array_equal(t[i,q],y):
        raise ValueError('inconsistent prefix class')
    return t

def search(prefix, classes, budgets=(1,8,32,128), seed=0, horizon=8, *, history=8):
    """Return best sampled unique future by exact joint mass under fair unseen bits.
    A zero-fill candidate is always included. The search is not an exact MAP
    unless it attains the lower bound imposed by the first frame.
    """
    if not budgets or min(budgets)<1 or horizon<1: raise ValueError('positive budgets')
    classes=canonical_classes(classes)
    t=prefix_tables(prefix,classes,history=history)
    rng=np.random.default_rng(seed)
    all_out={k:[] for k in budgets}
    for start in range(0,len(t),32):
        tt=t[start:start+32]
        b=len(tt); k=max(budgets)
        unknown=tt<0
        assignment=rng.integers(0,2,(b,k,tt.shape[1]),dtype=np.uint8)
        assignment[:,0]=0
        if k>1: assignment[:,1]=1
        assignment=np.where(unknown[:,None,:],assignment,tt[:,None,:]).astype(np.uint8)
        cur=np.broadcast_to(prefix[start:start+b,-1,None],(b,k,*prefix.shape[-2:]))
        used=np.zeros((b,k),np.uint64)
        rows=np.arange(b)[:,None,None,None]
        cand=np.arange(k)[None,:,None,None]
        frames=[]
        for step in range(horizon):
            q=classes[patterns(cur)]
            bits=np.where(unknown[np.arange(b)[:,None,None,None],q],np.left_shift(np.uint64(1),q.astype(np.uint64)),np.uint64(0))
            used|=np.bitwise_or.reduce(bits.reshape(b,k,-1),axis=-1)
            if step==0: lower=np.bitwise_count(used[:,0])
            cur=assignment[rows,cand,q]
            frames.append(cur)
        trajectory=np.stack(frames,axis=2)
        count=np.bitwise_count(used)
        for budget in budgets:
            j=count[:,:budget].argmin(1)
            idx=np.arange(b)
            cost=count[idx,j]
            all_out[budget].append((trajectory[idx,j],cost,cost==lower))
    return {k:tuple(np.concatenate([v[j] for v in vals]) for j in range(3)) for k,vals in all_out.items()}
