"""Constructive CNN (momentum-induction testbed), docs/SETTINGS.md §6.

Network A (rule evidence, one forward per sequence):
  featurizer Conv3d(1->3,(2,3,3)) -> split Linear+ReLU -> conjoin Linear+ReLU
  -> gate -> spatial aggregation -> temporal merge -> log1p -> condition(36).
Network B (per-step context code):
  featurizer Conv2d(1->2,3x3) -> split -> conjoin -> code(18).
Head (per cell, bilinear): W_c(code) * W_e(cond) -> ReLU -> 1x1 conv -> logit.

Arms:
  existence  — analytic featurizer/split/conjoin (exact slot detectors),
               aggregation fixed to plain counting; only the head is trained
               (2,017 trainable params).
  emergence  — same structure, all weights learned from standard init
               (4,650 total params at 8x8).

Analytic detector construction (±1 inputs; black = +1):
  featurizer channels: center_t, sum9_t (sum of the 3x3 block at frame t),
  center_{t+1}. sum9 = 2k-9 for k live cells in the block, odd values in
  {-9..9}; center of an isolated-live-cell context (s=1, n) gives
  v(s,n) = 2n-7, and (s=0, n) gives 2n-9.
  split units: p_v = ReLU(m - v), q_v = ReLU(v - m) for the 10 odd values v
  (so p_v + q_v = |m - v| >= 2 whenever m != v), plus s_pos/s_neg and o_pos/o_neg
  (sign indicators of the two centers).
  conjoin slot(s,n,o) = ReLU(s_x + o_x - p_v - q_v - 1)  -> exactly 1 on the
  matching (window -> outcome) triple, 0 otherwise.
  B-side conjoin code(s,n) = ReLU(s_x - p_v - q_v).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_ODD_VALUES = list(range(-9, 10, 2))  # -9,-7,...,9


def _circ3d(x, p):
    return F.pad(x, (p, p, p, p, 0, 0), mode="circular")


def _circ2d(x, p):
    return F.pad(x, (p, p, p, p), mode="circular")


class SumPool2(nn.Module):
    """2x2 sum-pooling over spatial dims of (B,C,T,H,W)."""

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "grid must be a power of two"
        return x.reshape(B, C, T, H // 2, 2, W // 2, 2).sum(dim=(4, 6))


def _v_of(s, n):
    return 2 * n - 7 if s else 2 * n - 9


def _build_analytic_a(feat, split, conjoin):
    with torch.no_grad():
        # featurizer: ch0 = center_t, ch1 = sum9_t, ch2 = center_{t+1}
        w = torch.zeros(3, 1, 2, 3, 3)
        w[0, 0, 0, 1, 1] = 1.0
        w[1, 0, 0, :, :] = 1.0
        w[2, 0, 1, 1, 1] = 1.0
        feat.weight.copy_(w)
        feat.bias.zero_()
        # split (3 -> 24): 20 hinge pairs on the sum channel + 4 sign units
        sw = torch.zeros(24, 3)
        sb = torch.zeros(24)
        for j, v in enumerate(_ODD_VALUES):
            sw[2 * j, 1] = 1.0
            sb[2 * j] = -v          # p_v = ReLU(m - v)
            sw[2 * j + 1, 1] = -1.0
            sb[2 * j + 1] = v       # q_v = ReLU(v - m)
        sw[20, 0] = 1.0   # s_pos
        sw[21, 0] = -1.0  # s_neg
        sw[22, 2] = 1.0   # o_pos
        sw[23, 2] = -1.0  # o_neg
        split.weight.copy_(sw[:, :, None, None, None])
        split.bias.copy_(sb)
        # conjoin (24 -> 36): slot channel = 2*(9s+n) + o
        cw = torch.zeros(36, 24)
        cb = torch.full((36,), -1.0)
        for s in (0, 1):
            for n in range(9):
                v = _v_of(s, n)
                pj = 2 * _ODD_VALUES.index(v)
                for o in (0, 1):
                    ch = 2 * (9 * s + n) + o
                    cw[ch, pj] = -1.0
                    cw[ch, pj + 1] = -1.0
                    cw[ch, 20 if s else 21] = 1.0
                    cw[ch, 22 if o else 23] = 1.0
        conjoin.weight.copy_(cw[:, :, None, None, None])
        conjoin.bias.copy_(cb)


def _build_analytic_b(feat, split, conjoin):
    with torch.no_grad():
        w = torch.zeros(2, 1, 3, 3)
        w[0, 0, 1, 1] = 1.0   # center
        w[1, 0, :, :] = 1.0   # sum9
        feat.weight.copy_(w)
        feat.bias.zero_()
        sw = torch.zeros(22, 2)
        sb = torch.zeros(22)
        for j, v in enumerate(_ODD_VALUES):
            sw[2 * j, 1] = 1.0
            sb[2 * j] = -v
            sw[2 * j + 1, 1] = -1.0
            sb[2 * j + 1] = v
        sw[20, 0] = 1.0   # s_pos
        sw[21, 0] = -1.0  # s_neg
        split.weight.copy_(sw[:, :, None, None])
        split.bias.copy_(sb)
        cw = torch.zeros(18, 22)
        for s in (0, 1):
            for n in range(9):
                v = _v_of(s, n)
                pj = 2 * _ODD_VALUES.index(v)
                ch = 9 * s + n
                cw[ch, pj] = -1.0
                cw[ch, pj + 1] = -1.0
                cw[ch, 20 if s else 21] = 1.0
        conjoin.weight.copy_(cw[:, :, None, None])
        conjoin.bias.zero_()


AGG_INITS = ("ones", "random", "positive")


def apply_agg_init(pyramid, temporal_w, temporal_b, agg_init):
    """E4.6 control for the emergence aggregation sub-network (pyramid depthwise
    convs + temporal merge). 'ones' = registered analytic counting init
    (bitwise-identical to every earlier run). 'random' = PyTorch default init
    kept for the pyramid convs; temporal merge drawn like an nn.Linear(7->1)
    (U(+-1/sqrt(7))). 'positive' = sign-preserving random: U(0.5, 1.5)
    weights, zero biases (isolates the non-negativity prior from the exact
    ones values). Call it LAST in __init__ so every other layer's init stays
    identical to the registered run with the same seed (paired control)."""
    assert agg_init in AGG_INITS, agg_init
    if agg_init == "ones":
        return  # constructors already applied ones_/zeros_ (registered path)
    with torch.no_grad():
        if agg_init == "random":
            for conv in pyramid:
                conv.reset_parameters()          # PyTorch default (kaiming_uniform)
            bound = 1.0 / math.sqrt(temporal_w.shape[1])
            temporal_w.uniform_(-bound, bound)
            temporal_b.uniform_(-bound, bound)
        else:
            for conv in pyramid:
                conv.weight.uniform_(0.5, 1.5)
                conv.bias.zero_()
            temporal_w.uniform_(0.5, 1.5)
            temporal_b.zero_()


class ConstructiveCNN(nn.Module):
    """head='concat' (default): per-cell MLP on [code(18) | cond(36)] -> 128 -> 64 -> 1.
    head='bilinear': W_c(code) * W_e(cond) -> out. Both registered settings
    (docs/SETTINGS.md §6); the bilinear default path lives on W_e's bias."""

    def __init__(self, grid=8, arm="emergence", head="concat",
                 det_feat=3, det_split=27, agg_init="ones"):
        super().__init__()
        assert arm in ("existence", "emergence")
        assert head in ("concat", "bilinear")
        assert grid & (grid - 1) == 0 and grid >= 4
        self.grid = grid
        self.arm = arm
        self.head_type = head
        # E6.6g detector redundancy (emergence only; 3/27 = registered, bitwise)
        self.det_feat, self.det_split = int(det_feat), int(det_split)
        # E4.6 aggregation init control (emergence only; 'ones' = registered)
        self.agg_init = agg_init

        # ---- Network A: featurizer / split / conjoin ----
        if arm == "existence":
            self.feat_a = nn.Conv3d(1, 3, kernel_size=(2, 3, 3))
            self.split_a = nn.Conv3d(3, 24, 1)
            self.conjoin_a = nn.Conv3d(24, 36, 1)
            _build_analytic_a(self.feat_a, self.split_a, self.conjoin_a)
            for m in (self.feat_a, self.split_a, self.conjoin_a):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
            self.gate_theta = None  # identity gate
        else:
            self.feat_a = nn.Conv3d(1, self.det_feat, kernel_size=(2, 3, 3))
            self.split_a = nn.Conv3d(self.det_feat, self.det_split, 1)
            self.conjoin_a = nn.Conv3d(self.det_split, 36, 1)
            self.gate_theta = nn.Parameter(torch.zeros(36))

        # ---- Network A: aggregation ----
        if arm == "existence":
            self.pyramid = None  # fixed global counting
        else:
            n_levels = int(torch.log2(torch.tensor(float(grid))).item()) - 1
            self.pyramid = nn.ModuleList(
                [nn.Conv3d(36, 36, kernel_size=(1, 3, 3), groups=36) for _ in range(n_levels)]
            )
            for conv in self.pyramid:
                nn.init.ones_(conv.weight)
                nn.init.zeros_(conv.bias)
            self.pool = SumPool2()
            self.temporal_w = nn.Parameter(torch.ones(36, 7))
            self.temporal_b = nn.Parameter(torch.zeros(36))

        # ---- Network B ----
        self.feat_b = nn.Conv2d(1, 2, 3)
        if arm == "existence":
            self.split_b = nn.Conv2d(2, 22, 1)
            self.conjoin_b = nn.Conv2d(22, 18, 1)
            _build_analytic_b(self.feat_b, self.split_b, self.conjoin_b)
            for m in (self.feat_b, self.split_b, self.conjoin_b):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        else:
            self.split_b = nn.Conv2d(2, 18, 1)
            self.conjoin_b = nn.Conv2d(18, 18, 1)

        # ---- Head ----
        if head == "concat":
            self.fc1 = nn.Conv2d(18 + 36, 128, 1)
            self.fc2 = nn.Conv2d(128, 64, 1)
            self.fc3 = nn.Conv2d(64, 1, 1)
        else:
            self.W_c = nn.Conv2d(18, 36, 1)
            self.W_e = nn.Linear(36, 36)
            self.out = nn.Conv2d(36, 1, 1)

        if arm == "emergence":
            apply_agg_init(self.pyramid, self.temporal_w, self.temporal_b, agg_init)

    # ---- forward pieces ----
    def forward_a(self, prefix_pm1):
        """prefix_pm1: (B,1,8,H,W) in ±1 -> condition (B,36)."""
        h = self.feat_a(_circ3d(prefix_pm1, 1))            # (B,3,7,H,W)
        h = F.relu(self.split_a(h))
        h = F.relu(self.conjoin_a(h))                      # slot indicators
        if self.arm == "existence":
            h = h.sum(dim=(3, 4))                          # (B,36,7) global counts
            cond = h.sum(dim=-1)                           # uniform temporal merge
        else:
            h = F.relu(h - self.gate_theta[None, :, None, None, None])
            for conv in self.pyramid:
                h = self.pool(conv(_circ3d(h, 1)))
            h = h.sum(dim=(3, 4))                          # (B,36,7)
            cond = torch.einsum("bct,ct->bc", h, self.temporal_w) + self.temporal_b
        return torch.log1p(cond.clamp(min=0))

    def forward_b(self, frames_pm1):
        """frames_pm1: (N,1,H,W) in ±1 -> context codes (N,18,H,W)."""
        h = self.feat_b(_circ2d(frames_pm1, 1))
        h = F.relu(self.split_b(h))
        return F.relu(self.conjoin_b(h))

    def head(self, codes, cond):
        """codes: (N,18,H,W); cond: (N,36) -> logits (N,H,W)."""
        if self.head_type == "concat":
            N, _, H, W = codes.shape
            e = cond[:, :, None, None].expand(N, 36, H, W)
            x = torch.cat([codes, e], dim=1)
            return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))).squeeze(1)
        c = self.W_c(codes)                                # (N,36,H,W)
        e = self.W_e(cond)[:, :, None, None]               # (N,36,1,1)
        return self.out(F.relu(c * e)).squeeze(1)

    def forward(self, frames_pm1):
        """Teacher-forced parallel prediction.

        frames_pm1: (B,1,16,H,W) ±1. Prefix = frames 0..7; inputs = frames
        7..14; returns logits (B,8,H,W) for frames 8..15.
        """
        B = frames_pm1.shape[0]
        cond = self.forward_a(frames_pm1[:, :, :8])
        ins = frames_pm1[:, :, 7:15].reshape(B * 8, 1, self.grid, self.grid)
        codes = self.forward_b(ins)
        cond_rep = cond.repeat_interleave(8, dim=0)
        logits = self.head(codes, cond_rep)
        return logits.reshape(B, 8, self.grid, self.grid)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """Autoregressive 8-step rollout. prefix_pm1: (B,1,8,H,W) ±1
        -> predictions (B,8,H,W) uint8."""
        B = prefix_pm1.shape[0]
        cond = self.forward_a(prefix_pm1)
        x = prefix_pm1[:, :, 7]                             # (B,1,H,W)
        preds = []
        for _ in range(8):
            codes = self.forward_b(x)
            logits = self.head(codes, cond)
            pred = (logits > 0).to(torch.uint8)
            preds.append(pred)
            x = (2.0 * pred.float() - 1.0).unsqueeze(1)
        return torch.stack(preds, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
