"""Diff-NCA-CFG — NCA-as-denoiser, config-adapted to our next-frame protocol
(R2 related-work baseline).

Source architecture: Kalkhof, Kuehn, Frisch, Mukhopadhyay, "Frequency-Time
Diffusion with Neural Cellular Automata" (arXiv:2401.06291; Diff-NCA model) —
the diffusion + cellular-automata generative framework: the UNet denoiser of a
DDPM is replaced by a one-cell NCA model that denoises purely by iterated
local communication.

Kept from the paper (its identity):
  - one-cell model: a learned 3x3 conv (local communication) followed by a
    per-cell 1x1-conv update stack, producing a RESIDUAL increment;
  - iterated NCA micro-steps per denoising call (perceptive range grows by 1
    per step; 4 steps cover the 8x8 torus);
  - channel layout: input channels + output channel + empty hidden channels
    carrying state between micro-steps;
  - sinusoidal embedding (diffusion timestep, NCA step) through an MLP,
    injected multiplicatively at two points of the cell update (their
    multiplicative conditioning blocks M_eb1 / M_eb2);
  - parameter count at the paper's scale (~0.37M-0.57M vs their 331k).

Config-adapted to our task (deviations, documented):
  - the denoiser conditions on the 8 clean prefix frames as extra input
    channels (the paper denoises unconditioned images; our task requires the
    prefix);
  - absolute (x, y) position embedding DROPPED: our CA family is
    translation-equivariant on a torus, so circular padding replaces both
    the bounded canvas and the positional signal;
  - canvas frames are denoised as independent 8x8 images (the paper is a 2D
    image model; our canvas is temporal), sharing one cell model;
  - predicts x0 logits under our single-step BCE protocol with the same
    VP-AR(1) schedule and reverse chain as DiffusionVanilla (the paper
    predicts noise with an L1+L2 loss) — the schedule/sampler is protocol,
    the NCA denoiser is the architecture;
  - hidden width tuned into the R2 parameter budget (0.5M-1.5M trainable).

Unlike DiffusionVanilla this model also accepts a SINGLE-frame canvas in
denoise() (xt (B,H,W) with hist (B,8,H,W)), which gives eval._tf_forward a
well-defined teacher-forced path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_vanilla import DiffusionVanilla

_SIN_H = 16          # sin/cos pairs per scalar
_EMB_IN = 4 * _SIN_H       # t_diff and nca-step, each sin+cos -> 64


class DiffNCACfg(DiffusionVanilla):
    def __init__(self, grid=8, arm="emergence", head="concat", t_per_frame=False,
                 n_channel=160, hidden=640, nca_steps=4, emb_e=8):
        nn.Module.__init__(self)   # skip the 3D-CNN stack of DiffusionVanilla
        self.grid = grid
        self.arm = arm
        self.head_type = head
        self.t_per_frame = bool(t_per_frame)
        self.n_channel = n_channel
        self.hidden = hidden
        self.nca_steps = nca_steps
        self.emb_e = emb_e
        self.in_ch = 9             # noised frame (1) + clean history (8)
        assert n_channel > self.in_ch
        freqs = torch.exp(torch.arange(_SIN_H, dtype=torch.float32)
                          * (-torch.log(torch.tensor(10000.0)) / _SIN_H))
        self.register_buffer("sin_freqs", freqs)                   # (_SIN_H,)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3)
        self.fc1 = nn.Conv2d(2 * n_channel, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, n_channel, 1)
        self.emb1 = nn.Linear(_EMB_IN, 256)
        self.emb2 = nn.Linear(256, emb_e)
        self.m_eb1 = nn.Linear(emb_e, 2 * n_channel)  # scale+shift on conv3 out
        self.m_eb2 = nn.Linear(emb_e, hidden)         # gate on fc1 out

    # -- cell ------------------------------------------------------------
    def _embed(self, t, step):
        """t: (M,) long diffusion level; step: int NCA micro-step -> (M,emb_e)."""
        f = self.sin_freqs
        tf = t.to(f.dtype).unsqueeze(1) * f                      # (M,_SIN_H)
        sf = float(step) * f.unsqueeze(0).expand_as(tf)
        feat = torch.cat([torch.sin(tf), torch.cos(tf),
                          torch.sin(sf), torch.cos(sf)], dim=1)  # (M,64)
        return self.emb2(F.silu(self.emb1(feat)))

    def _update(self, state, e):
        y = self.conv3(F.pad(state, (1, 1, 1, 1), mode="circular"))
        scale, shift = self.m_eb1(F.silu(e)).chunk(2, dim=1)
        y = y * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) \
            + shift.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.fc1(torch.cat([y, state], dim=1)))
        h = h * self.m_eb2(F.silu(e)).unsqueeze(-1).unsqueeze(-1)
        return state + self.fc2(h)

    # -- denoiser interface (DiffusionVanilla protocol) --------------------
    def denoise(self, xt, hist, t):
        """xt: (B,8,H,W) noised canvas or (B,H,W) single noised frame;
        hist: (B,8,H,W) clean prefix; t: (B,) or (B,8) long levels.
        Returns (u, u) logits shaped like xt (second = interface parity)."""
        single = xt.dim() == 3
        if single:
            xt4 = xt.unsqueeze(1)                                # (B,1,H,W)
        else:
            xt4 = xt
        B, K, H, W = xt4.shape
        M = B * K
        x = xt4.reshape(M, 1, H, W)
        hst = (hist.unsqueeze(1).expand(B, K, 8, H, W).reshape(M, 8, H, W)
               if K > 1 else hist)
        if t.dim() == 2:
            t_m = t.reshape(M)
        elif K > 1:
            t_m = t.repeat_interleave(K)
        else:
            t_m = t
        state = x.new_zeros(M, self.n_channel, H, W)
        state[:, :self.in_ch] = torch.cat([x, hst], dim=1)
        for k in range(self.nca_steps):
            state = self._update(state, self._embed(t_m, k))
        u = state[:, self.in_ch]                                 # (M,H,W)
        u = u.reshape(B, K, H, W)
        if single:
            u = u.squeeze(1)
        return u, u


def build(**kwargs) -> DiffNCACfg:
    """Factory for the trainer registry."""
    return DiffNCACfg(**kwargs)
