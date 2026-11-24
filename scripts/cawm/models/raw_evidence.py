"""Experimental prefix-conditioned CA: learned raw-pattern evidence matching.

This is a structured alternative, NOT the original R0 architecture. The
nine bits are supplied but neither neighbour count nor hidden rule is given.
A learned kernel aggregates signed historical observations, with an optional
learned observation-noise floor. Keep the inferred rule fixed during rollout.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RawEvidenceCA(nn.Module):
    def __init__(self, grid=8, shifted=False, mode="count"):
        super().__init__()
        if mode not in ("count", "mean", "sqrt", "support"):
            raise ValueError(mode)
        self.grid, self.shifted, self.mode = grid, shifted, mode
        self.enc = nn.Sequential(nn.Linear(9,16), nn.Tanh(), nn.Linear(16,8))
        self.log_gain = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))
        if mode != "mean":
            self.noise_logit = nn.Parameter(torch.tensor(-4.))
        if mode == "support":
            self.log_prior_count = nn.Parameter(torch.tensor(0.))
        self.register_buffer("bits", ((torch.arange(512)[:,None] >> torch.arange(9)) & 1).float())

    @staticmethod
    def pattern(x):
        """0/1 cells, arbitrary leading axes, toroidal final two axes."""
        x = x.long()
        return sum(torch.roll(x, (dy,dx), (-2,-1)) * (1 << i)
                   for i,(dy,dx) in enumerate((a,b) for a in (-1,0,1) for b in (-1,0,1)))

    def evidence(self, prefix):
        """Only observed frames; prefix shape (B,8,H,W), 0/1."""
        if prefix.ndim != 4 or prefix.shape[1] != 8 or prefix.shape[-2] != prefix.shape[-1] or prefix.shape[-1] < 3:
            raise ValueError("expected 8 observed frames on a square grid >=3")
        p = self.pattern(prefix[:,:7]).flatten(1)
        y = prefix[:,1:].flatten(1).float()
        c = y.new_zeros((len(y),512))
        z = torch.zeros_like(c)
        c.scatter_add_(1,p,torch.ones_like(y))
        z.scatter_add_(1,p,2*y-1)
        return c,z

    def kernel(self):
        h = self.enc(self.bits)
        d = (h.square().sum(1)[:,None]+h.square().sum(1)[None,:]-2*h@h.T).clamp_min(0)
        return (-d).exp()

    def table(self,c,z):
        w = self.kernel()
        signed = z@w.T
        if self.mode == "mean":
            return F.softplus(self.log_gain)*signed/(c@w.T).clamp_min(1e-8)+self.bias
        # More independent agreeing observations provide greater confidence.
        # A trainable symmetric flip probability bounds next-cell confidence.
        if self.mode == "sqrt":
            signed = signed / (c @ w.T).clamp_min(1.).sqrt()
        if self.mode == "support":
            # Suppress unsupported extrapolation; strong evidence saturates.
            signed = signed / (c @ w.T + F.softplus(self.log_prior_count) + 1e-4)
        score = F.softplus(self.log_gain)*signed+self.bias
        eps = .5*self.noise_logit.sigmoid()
        prob = eps+(1-2*eps)*score.sigmoid()
        return torch.logit(prob,eps=1e-6)

    def query(self,prev):
        q = self.pattern(prev)
        return torch.roll(q,(q.shape[-2]//2,q.shape[-1]//2),(-2,-1)) if self.shifted else q

    @staticmethod
    def gather(table,q):
        return table.gather(1,q.flatten(1)).reshape(q.shape)

    def first(self,prefix_pm1):
        prefix = (prefix_pm1[:,0]>0).long()
        return self.gather(self.table(*self.evidence(prefix)),self.query(prefix[:,7]))

    def forward(self,frames_pm1):
        """Eight-frame teacher-forced likelihood; rule inferred only at prefix."""
        x = (frames_pm1[:,0]>0).long()
        t = self.table(*self.evidence(x[:,:8]))
        return torch.stack([self.gather(t,self.query(x[:,i])) for i in range(7,15)],1)

    @torch.no_grad()
    def rollout(self,prefix_pm1,steps=8):
        if not isinstance(steps,int) or isinstance(steps,bool) or steps<1:
            raise ValueError("steps must be a positive integer")
        x = (prefix_pm1[:,0]>0).long()
        t = self.table(*self.evidence(x))
        cur=x[:,7];out=[]
        for _ in range(steps):
            cur=(self.gather(t,self.query(cur))>0).long()
            out.append(cur.to(torch.uint8))
        return torch.stack(out,1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
