"""R0-CNN baseline (E4.2): a plain conv net over the 9 stacked frames.

Input = 9 channels (8 history frames + current frame), 4x Conv2d(->64, 3x3,
circular pad) + ReLU, then a 1x1 conv -> 1 logit per cell. No detector stack,
no condition vector, no attention — the generic-CNN control for the ladder
(docs/CLAIMS_AND_INFOFLOW.md §3.1 rung R0). Forward/rollout interface matches
ConstructiveCNN so train.py and eval.py treat it identically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constructive_cnn import _circ2d

CH = 64


class ARCNN(nn.Module):
    def __init__(self, grid=8, arm="emergence", head="concat"):
        super().__init__()
        assert grid & (grid - 1) == 0 and grid >= 4
        self.grid = grid
        self.arm = arm
        self.head_type = head
        self.c1 = nn.Conv2d(9, CH, 3)
        self.c2 = nn.Conv2d(CH, CH, 3)
        self.c3 = nn.Conv2d(CH, CH, 3)
        self.c4 = nn.Conv2d(CH, CH, 3)
        self.out = nn.Conv2d(CH, 1, 1)

    def _step(self, x9):
        """x9: (N,9,H,W) ±1 -> logits (N,H,W)."""
        h = F.relu(self.c1(_circ2d(x9, 1)))
        h = F.relu(self.c2(_circ2d(h, 1)))
        h = F.relu(self.c3(_circ2d(h, 1)))
        h = F.relu(self.c4(_circ2d(h, 1)))
        return self.out(h).squeeze(1)

    def forward(self, frames_pm1):
        """Teacher-forced parallel prediction.
        frames_pm1: (B,1,16,H,W) ±1 -> logits (B,8,H,W) for frames 8..15.
        For output frame 8+j the input is the 8 frames j..j+7 (history) with
        the current frame j+7 repeated -> 9 channels."""
        B = frames_pm1.shape[0]
        f = frames_pm1[:, 0]                                 # (B,16,H,W)
        outs = []
        for j in range(8):
            x9 = torch.cat([f[:, j:j + 8], f[:, j + 7:j + 8]], dim=1)  # (B,9,H,W)
            outs.append(self._step(x9))
        return torch.stack(outs, dim=1)                      # (B,8,H,W)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """Autoregressive 8-step rollout. prefix_pm1: (B,1,8,H,W) ±1
        -> predictions (B,8,H,W) uint8."""
        hist = [prefix_pm1[:, 0, i] for i in range(8)]       # 8 x (B,H,W) ±1
        preds = []
        for _ in range(8):
            x9 = torch.stack(hist[-8:] + [hist[-1]], dim=1)  # (B,9,H,W)
            logits = self._step(x9)
            pred = (logits > 0).to(torch.uint8)
            preds.append(pred)
            hist.append(2.0 * pred.float() - 1.0)
        return torch.stack(preds, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
