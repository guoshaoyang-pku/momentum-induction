"""Persistent latent rule bits with independent observation flips.

Class partition is frozen from training histories; noise rate fitted using
training prefixes only. Likelihood integrates over each latent bit once,
including repeated queries in the same frame and across time.
"""
import numpy as np
from .rule_hypotheses import raw_pattern

def counts(frames, classes):
    x=np.asarray(frames); c=np.asarray(classes)
    if x.ndim!=4 or x.shape[1]<2 or min(x.shape[2:])<3 or not np.isin(x,[0,1]).all():
        raise ValueError("binary B,T,H,W with T>=2 required")
    if c.shape!=(512,) or not np.issubdtype(c.dtype,np.integer) or c.min()<0 or c.max()>=512:
        raise ValueError("invalid classes")
    k=int(c.max())+1
    q=c[raw_pattern(x[:,:-1])].reshape(len(x),-1)
    y=x[:,1:].reshape(len(x),-1)
    key=(np.arange(len(x))[:,None]*k+q).ravel()
    n=np.bincount(key,minlength=len(x)*k).reshape(len(x),k)
    a=np.bincount(key,weights=y.ravel(),minlength=len(x)*k).reshape(len(x),k)
    return n-a,a

def log_weights(z,o,p):
    if not np.isfinite(p) or not 0<=p<=.5:raise ValueError("noise in [0,.5]")
    if p==0:
        return np.where(o==0,-np.log(2),-np.inf),np.where(z==0,-np.log(2),-np.inf)
    a=np.log1p(-p);b=np.log(p)
    return z*a+o*b-np.log(2),o*a+z*b-np.log(2)

def evidence(z,o,p):
    a,b=log_weights(z,o,p)
    return np.logaddexp(a,b)

def posterior(prefix,classes,p):
    if np.asarray(prefix).ndim!=4 or prefix.shape[1]!=8:raise ValueError("only 8 observed frames")
    z,o=counts(prefix,classes)
    a,b=log_weights(z,o,p)
    denom=np.logaddexp(a,b)
    if not np.isfinite(denom).all():raise ValueError("impossible history under this model")
    return np.exp(b-denom)

def fit_noise(prefix,classes):
    """Grid MLE on independent uniform latent bits; not supplied true p."""
    if np.asarray(prefix).ndim!=4 or prefix.shape[1]!=8:raise ValueError("only training histories")
    z,o=counts(prefix,classes)
    # Aggregate repeated count pairs for efficient deterministic search.
    pairs,n=np.unique(np.stack([z,o],axis=-1).reshape(-1,2),axis=0,return_counts=True)
    z,o=pairs.T
    def score(p):return float(np.dot(evidence(z,o,p),n))
    grid=np.linspace(0,.5,501)
    p=float(grid[np.argmax([score(v) for v in grid])])
    fine=np.linspace(max(0,p-.001),min(.5,p+.001),21)
    return float(fine[np.argmax([score(v) for v in fine])])

def sequence_scores(frames,classes,p):
    """Exact joint conditional NLL per frame, plus independent pixel marginals.

    Targets are only used for scoring/conditioning later teacher-forced steps,
    never for the separate autonomous rollout.
    """
    x=np.asarray(frames)
    if x.ndim!=4 or x.shape[1]<9:raise ValueError("need observed prefix and target")
    z,o=counts(x[:,:8],classes)
    a,b=log_weights(z,o,p);prev=np.logaddexp(a,b)
    if not np.isfinite(prev).all():raise ValueError("impossible prefix")
    joint=[];marg=[];pixel=[]
    for i in range(8,x.shape[1]):
        q=np.asarray(classes)[raw_pattern(x[:,i-1])]
        with np.errstate(invalid="ignore"):
            post=np.exp(b-prev)
        post=np.where(np.isfinite(post),post,.5)
        prob=p+(1-2*p)*np.take_along_axis(post,q.reshape(len(x),-1),axis=1)
        y=x[:,i].reshape(len(x),-1)
        with np.errstate(divide="ignore"):
            marg.append(-np.log(np.where(y,prob,1-prob)).sum(1))
        pixel.append(((prob>.5)==y).mean(1))
        dz,do=counts(x[:,i-1:i+1],classes)
        z+=dz;o+=do;a,b=log_weights(z,o,p)
        now=np.logaddexp(a,b)
        # Once impossible the full sequence remains impossible. First failure
        # is +inf; later conditionals are undefined (NaN), not finite scores.
        with np.errstate(invalid="ignore"):
            joint.append((prev-now).sum(1))
        prev=now
    return np.stack(joint,1),np.stack(marg,1),np.stack(pixel,1)

def rollout(prefix,classes,p,seed=None,sample=False,steps=8):
    if steps<1:raise ValueError("positive steps")
    post=posterior(prefix,classes,p)
    rng=np.random.default_rng(seed)
    table=(rng.random(post.shape)<post) if sample else (post>.5)
    cur=np.asarray(prefix)[:,-1].copy();frames=[]
    for _ in range(steps):
        q=np.asarray(classes)[raw_pattern(cur)]
        cur=table[np.arange(len(cur))[:,None,None],q].astype(np.uint8)
        if sample:cur^=(rng.random(cur.shape)<p).astype(np.uint8)
        frames.append(cur)
    return np.stack(frames,1)
