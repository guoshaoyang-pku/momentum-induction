"""Independent Markov flips of latent rule bits, distinct from pixel noise.

A bit is shared by every cell querying its class within one transition.
Between successive transitions it flips with probability q. Independent
output-pixel flips have probability p. Exact forward filtering conditional
on observed states uses per-transition class counts; latent bits factorize.
"""
import numpy as np
from .noisy_posterior import counts
from .rule_hypotheses import raw_pattern

def _rates(rates):
    r=np.asarray(rates,dtype=float)
    if r.ndim!=2 or r.shape[1]!=2 or not len(r) or not np.isfinite(r).all() or np.any((r<0)|(r>.5)):
        raise ValueError("rates must be rows (p,q) in [0,.5]")
    return r

def _observations(frames,classes):
    x=np.asarray(frames)
    if x.ndim!=4 or x.shape[1]<2 or not len(x):
        raise ValueError("nonempty binary trajectories required")
    return [counts(x[:,t:t+2],classes) for t in range(x.shape[1]-1)]

def _update(post,z,o,p,q,first):
    # post axes = candidate rate, trajectory, class.
    pred=post if first else q+(1-2*q)*post
    with np.errstate(divide="ignore",invalid="ignore"):
        lp=np.log(p); lc=np.log1p(-p)
        # Avoid 0 * log(0).
        def emission(a,b):
            return np.where(a==0,0,a*lc)+np.where(b==0,0,b*lp)
        a=np.log1p(-pred)+emission(z,o)
        b=np.log(pred)+emission(o,z)
        norm=np.logaddexp(a,b)
        out=np.exp(b-norm)
    return np.where(np.isfinite(out),out,.5),norm.sum(axis=2)

def _filter(obs,rates):
    r=_rates(rates)
    p=r[:,0,None,None];q=r[:,1,None,None]
    post=np.full((len(r),*obs[0][0].shape),.5)
    loglik=np.zeros(post.shape[:2])
    for t,(z,o) in enumerate(obs):
        post,inc=_update(post,z[None],o[None],p,q,t==0)
        loglik+=inc
    return post,loglik

def fit_rates(prefix,classes,candidates,batch=64):
    """Grid MLE, shared population rates; training prefix only."""
    if np.asarray(prefix).ndim!=4 or prefix.shape[1]!=8 or batch<1:
        raise ValueError("eight observed frames and positive batch required")
    r=_rates(candidates);scores=np.zeros(len(r))
    for i in range(0,len(prefix),batch):
        _,ll=_filter(_observations(prefix[i:i+batch],classes),r)
        scores+=ll.sum(1)
    if not np.isfinite(scores).any():
        raise ValueError("all rates impossible")
    return tuple(r[np.argmax(scores)]),scores

def posterior(prefix,classes,p,q):
    if np.asarray(prefix).ndim!=4 or prefix.shape[1]!=8:
        raise ValueError("only eight observed frames")
    post,ll=_filter(_observations(prefix,classes),[(p,q)])
    if not np.isfinite(ll).all():raise ValueError("impossible prefix")
    return post[0]

def sequence_scores(frames,classes,p,q):
    """Joint conditional NLL per transition, not independent pixel BCE.
    Impossible target yields +inf; following conditionals are undefined.
    """
    x=np.asarray(frames)
    if x.ndim!=4 or x.shape[1]<9:raise ValueError("prefix and targets required")
    post=posterior(x[:,:8],classes,p,q)[None]
    p,q=(np.array([[[v]]]) for v in (p,q))
    alive=np.ones(len(x),dtype=bool);loss=[]
    for z,o in _observations(x[:,7:],classes):
        post,inc=_update(post,z[None],o[None],p,q,False)
        loss.append(np.where(alive,-inc[0],np.nan))
        alive &= np.isfinite(inc[0])
    return np.stack(loss,1)

def rollout(prefix,classes,p,q,seed,steps=8):
    """True autonomous generation; sample one table then mutate it."""
    if steps<1:raise ValueError("positive steps")
    post=posterior(prefix,classes,p,q)
    rng=np.random.default_rng(seed)
    table=rng.random(post.shape)<post
    cur=np.asarray(prefix)[:,-1].copy();out=[]
    for _ in range(steps):
        table ^= rng.random(table.shape)<q
        query=np.asarray(classes)[raw_pattern(cur)]
        cur=table[np.arange(len(cur))[:,None,None],query].astype(np.uint8)
        cur ^= (rng.random(cur.shape)<p).astype(np.uint8)
        out.append(cur)
    return np.stack(out,1)
