"""Learn a local rule similarity metric from prefix transitions, not rule labels."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OFFSETS = [(dy, dx) for dy in (-1,0,1) for dx in (-1,0,1)]

def patterns(x):
    x = np.asarray(x, dtype=np.int64)
    return sum((1<<i)*np.roll(x, (dy,dx), axis=(-2,-1))
               for i,(dy,dx) in enumerate(OFFSETS))

def evidence(prefix):
    if prefix.ndim != 4 or prefix.shape[1] != 8 or not np.isin(prefix,[0,1]).all():
        raise ValueError("exactly eight observed binary frames required")
    q = patterns(prefix[:,:7]).reshape(len(prefix),-1)
    y = prefix[:,1:].reshape(len(prefix),-1).astype(np.int64)
    ix = (512*np.arange(len(prefix))[:,None]+q).ravel()
    shape = (len(prefix),512)
    return (np.bincount(ix,minlength=np.prod(shape)).reshape(shape).astype(np.float32),
            np.bincount(ix,weights=(2*y-1).ravel(),minlength=np.prod(shape)).reshape(shape).astype(np.float32))

def query(frame, shift=False):
    q = patterns(frame)
    return np.roll(q,(4,4),axis=(-2,-1)) if shift else q

class LocalMetric(nn.Module):
    def __init__(self, rank, frozen=None):
        super().__init__()
        self.encoder = nn.Linear(9,rank,bias=False)
        self.gain = nn.Parameter(torch.tensor(2.))
        self.bias = nn.Parameter(torch.tensor(0.))
        self.register_buffer('raw', ((torch.arange(512)[:,None]>>torch.arange(9))&1).float()*2-1)
        if frozen is not None:
            with torch.no_grad(): self.encoder.weight.copy_(torch.as_tensor(frozen))
            self.encoder.weight.requires_grad_(False)
    def table(self,c,z):
        a = self.encoder(self.raw)
        k = (-(a[:,None,:]-a[None,:,:]).square().sum(-1)).exp()
        return F.softplus(self.gain)*(z@k.T)/(c@k.T).clamp_min(1e-6)+self.bias

def rollout(table,prefix,shift=False,horizon=8):
    if prefix.ndim != 4 or prefix.shape[1]!=8: raise ValueError("prefix only")
    cur=prefix[:,-1].copy()
    out=[]
    for _ in range(horizon):
        q=query(cur,shift)
        cur=(np.take_along_axis(table,q.reshape(len(q),-1),axis=1).reshape(q.shape)>0).astype(np.uint8)
        out.append(cur)
    return np.stack(out,axis=1)
