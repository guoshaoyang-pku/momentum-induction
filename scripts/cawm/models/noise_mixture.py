"""Trajectory-level latent noise rate, persistent rule bits.

The rate is sampled once per world, not once per transition. Fit a population
prior on training prefixes; integrate the rate and rule bits for joint NLL.
"""
import numpy as np
from .noisy_posterior import counts, log_weights
from .rule_hypotheses import raw_pattern

def _lse(x,axis):
    return np.logaddexp.reduce(x,axis=axis)

def _validate(grid,weights):
    grid=np.asarray(grid,dtype=float); weights=np.asarray(weights,dtype=float)
    if (grid.ndim!=1 or not len(grid) or grid.shape!=weights.shape or
        not np.isfinite(grid).all() or not np.isfinite(weights).all() or
        np.any(grid<0) or np.any(grid>.5) or np.any(np.diff(grid)<=0) or
        np.any(weights<0) or not np.isclose(weights.sum(),1)):
        raise ValueError("invalid noise prior")
    return grid,weights

def _state(z,o,grid,weights):
    a,b=zip(*(log_weights(z,o,p) for p in grid))
    a=np.stack(a,1);b=np.stack(b,1)
    den=np.logaddexp(a,b)
    with np.errstate(divide="ignore",invalid="ignore"):
        logmass=den.sum(2)+np.log(weights)
        bit=np.exp(b-den)
    return logmass,np.where(np.isfinite(bit),bit,.5)

def _prefix(prefix,classes,grid,weights):
    if np.asarray(prefix).ndim!=4 or prefix.shape[1]!=8:
        raise ValueError("only eight observed frames")
    grid,weights=_validate(grid,weights)
    z,o=counts(prefix,classes)
    mass,bit=_state(z,o,grid,weights)
    den=_lse(mass,1)
    if not np.isfinite(den).all():raise ValueError("impossible prefix")
    return z,o,mass,bit,den

def fit_prior(prefix,classes,grid,iterations=100,tol=1e-8):
    """EM on training prefixes only; return fitted prior and likelihood trace."""
    grid=np.asarray(grid,dtype=float)
    w=np.full(len(grid),1/len(grid))
    z,o,mass,_,_=_prefix(prefix,classes,grid,w)
    ll=mass-np.log(w)
    trace=[]
    for _ in range(iterations):
        with np.errstate(divide="ignore"):
            m=ll+np.log(w)
        den=_lse(m,1)
        trace.append(float(den.sum()))
        new=np.exp(m-den[:,None]).mean(0)
        if np.max(abs(new-w))<=tol:
            w=new;break
        w=new
    with np.errstate(divide="ignore"):
        trace.append(float(_lse(ll+np.log(w),1).sum()))
    if np.min(np.diff(trace))<-1e-7:raise RuntimeError("EM likelihood decreased")
    return w,trace

def posterior(prefix,classes,grid,weights):
    _,_,mass,bit,den=_prefix(prefix,classes,grid,weights)
    return np.exp(mass-den[:,None]),bit

def sequence_scores(frames,classes,grid,weights):
    """Exact joint conditional NLL, not product of marginal pixel likelihoods."""
    x=np.asarray(frames)
    if x.ndim!=4 or x.shape[1]<9:raise ValueError("need prefix and targets")
    z,o,_,_,prev=_prefix(x[:,:8],classes,grid,weights)
    joint=[]
    for i in range(8,x.shape[1]):
        dz,do=counts(x[:,i-1:i+1],classes)
        z+=dz;o+=do
        mass,_=_state(z,o,grid,weights)
        now=_lse(mass,1)
        with np.errstate(invalid="ignore"):
            joint.append(prev-now)
        prev=now
    return np.stack(joint,1)

def rollout(prefix,classes,grid,weights,seed,steps=8):
    """Sample one shared rate AND one rule table for each trajectory."""
    if steps<1:raise ValueError("positive steps")
    w,bit=posterior(prefix,classes,grid,weights)
    rng=np.random.default_rng(seed)
    idx=(rng.random(len(w))[:,None]>=np.cumsum(w,1)[:,:-1]).sum(1)
    p=np.asarray(grid)[idx]
    table=rng.random(bit.shape[::2])<bit[np.arange(len(w)),idx]
    cur=np.asarray(prefix)[:,-1].copy();out=[]
    for _ in range(steps):
        q=np.asarray(classes)[raw_pattern(cur)]
        cur=table[np.arange(len(cur))[:,None,None],q].astype(np.uint8)
        cur^=(rng.random(cur.shape)<p[:,None,None]).astype(np.uint8)
        out.append(cur)
    return np.stack(out,1),p
