"""NCA-CFG — Growing-NCA-style recurrent per-cell model, config-adapted to
our next-frame protocol (R2 related-work baseline).

Source architecture: Mordvintsev, Randazzo, Niklasson, Levin, "Growing
Neural Cellular Automata" (Distill, 2020) — the CA-as-generative-model work
that learns CA-like local dynamics to grow images from a seed.

Kept from the paper (its identity):
  - per-cell n_channel continuous state; a single seed/context plus hidden
    channels carry all memory;
  - fixed perception stage: identity + Sobel-x + Sobel-y 3x3 depthwise
    kernels (no learned perception weights);
  - shared per-cell update MLP (1x1 convs) producing a RESIDUAL increment;
  - stochastic per-cell update mask (fire_rate) during training;
  - iterative micro-stepping between readouts.

Config-adapted to our task (deviations, documented):
  - initialised from the last two observed frames (visible channels 0/1)
    instead of a single seed pixel, so the model can in principle sense
    momentum; hidden channels start at zero;
  - circular padding (our CA family lives on a torus; the paper grows on a
    bounded canvas);
  - hidden width scaled to land inside the R2 parameter budget
    (0.5M-1.5M trainable);
  - teacher-forced training: after predicting frame 8+j the visible channels
    are overwritten with the TRUE frames (hidden channels persist), matching
    our one-step tf convention; at rollout the model's own thresholded
    predictions are fed back instead;
  - inference uses deterministic updates (fire_rate=1.0): with per-cell
    stochastic masks an exact frame match is impossible by construction,
    which would make the baseline trivially zero rather than informative.

Interface is identical to `scripts/cawm/models/arcnn.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]]) / 8.0
_SOBEL_Y = _SOBEL_X.t().contiguous()
_IDENTITY = torch.tensor([[0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]])


class NCAConfigWM(nn.Module):
    def __init__(self, grid: int = 8, n_channel: int = 128, hidden: int = 1024,
                 steps_per_frame: int = 2, fire_rate: float = 0.5):
        super().__init__()
        self.grid = grid
        self.n_channel = n_channel
        self.steps_per_frame = steps_per_frame
        self.fire_rate = fire_rate
        kernels = torch.stack([_IDENTITY, _SOBEL_X, _SOBEL_Y])       # (3,3,3)
        # depthwise: n_channel groups, 3 kernels per channel
        weight = kernels.unsqueeze(1).repeat(n_channel, 1, 1, 1)     # (3C,1,3,3)
        self.register_buffer("perception_kernels", weight)
        self.fc1 = nn.Conv2d(3 * n_channel, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, n_channel, 1)

    # -- core ------------------------------------------------------------
    def _perceive(self, state):
        """state: (B,C,H,W) -> (B,3C,H,W), circular padding (torus)."""
        x = F.pad(state, (1, 1, 1, 1), mode="circular")
        return F.conv2d(x, self.perception_kernels, groups=self.n_channel)

    def _update(self, state, training: bool):
        delta = self.fc2(F.relu(self.fc1(self._perceive(state))))
        if training and self.fire_rate < 1.0:
            mask = (torch.rand_like(state[:, :1]) < self.fire_rate).to(state.dtype)
            delta = delta * mask
        return state + delta

    def _micro_steps(self, state, training: bool):
        for _ in range(self.steps_per_frame):
            state = self._update(state, training)
        return state

    def _init_state(self, f_prev, f_curr):
        """f_prev/f_curr: (B,H,W) +-1 -> state (B,C,H,W)."""
        B = f_prev.shape[0]
        state = f_prev.new_zeros(B, self.n_channel, self.grid, self.grid)
        state[:, 0] = f_prev
        state[:, 1] = f_curr
        return state

    @staticmethod
    def _readout(state):
        """Visible channel 1 (the "current frame" slot) as logits (B,H,W)."""
        return state[:, 1]

    # -- trainer interface -------------------------------------------------
    def forward(self, frames_pm1):
        """(B,1,16,H,W) +-1 -> logits (B,8,H,W) for frames 8..15.

        Teacher-forced: after each prediction the visible channels are
        overwritten with the true frames; hidden channels persist across the
        8 output steps, so the model learns to use them as memory.
        """
        f = frames_pm1[:, 0]                                  # (B,16,H,W)
        state = self._init_state(f[:, 6], f[:, 7])
        outs = []
        for j in range(8):
            state = self._micro_steps(state, self.training)
            outs.append(self._readout(state))
            state = state.clone()
            state[:, 0] = f[:, 7 + j]
            state[:, 1] = f[:, 8 + j]
        return torch.stack(outs, dim=1)                       # (B,8,H,W)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """(B,1,8,H,W) +-1 -> (B,8,H,W) uint8, autoregressive 8-step.

        Deterministic updates at inference (fire_rate bypassed); the model's
        own thresholded predictions become the new visible frame.
        """
        f = prefix_pm1[:, 0]
        state = self._init_state(f[:, 6], f[:, 7])
        preds = []
        prev = f[:, 7]
        for _ in range(8):
            state = self._micro_steps(state, training=False)
            logits = self._readout(state)
            bits = logits > 0
            preds.append(bits.to(torch.uint8))
            nxt = bits.to(state.dtype).mul(2.0).sub(1.0)
            state = state.clone()
            state[:, 0] = prev
            state[:, 1] = nxt
            prev = nxt
        return torch.stack(preds, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())


def build(**kwargs) -> NCAConfigWM:
    """Factory for the trainer registry."""
    return NCAConfigWM(**kwargs)


try:  # self-register with the training entry point when importable
    from ..train import register_model

    register_model("nca_cfg", build)
except Exception:  # pragma: no cover - registry optional at import time
    pass
