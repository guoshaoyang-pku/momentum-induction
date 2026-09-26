"""R0-diffusion baseline (E4.3): a generic 3D-conv denoiser — no constructive
condition, no lookup cascade, no detector stacks. The generic-DDPM control for
the ladder (docs/CLAIMS_AND_INFOFLOW.md §3.1 rung R0): same VP-AR(1) schedule,
same single-step BCE training loss, same 50-step posterior reverse chain as
DiffusionModel, but the denoiser is a plain 3D CNN.

Input = concat(canvas 8 frames, clean history 8 frames) along time ->
(B,1,16,H,W), plus 2 condition channels (a_t, sigma_t) -> 4x
Conv3d(->32,(3,3,3), circular spatial + zero temporal pad) + ReLU ->
Conv3d(32->1,(1,1,1)) logits for the 8 canvas frames.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import (SIGMA1, T_STEPS, alpha_t, chain_levels, sigma_t,
                        stratified_t)

CH = 32


def _circ3d_spatial(x, p):
    return F.pad(x, (p, p, p, p, 0, 0), mode="circular")


class DiffusionVanilla(nn.Module):
    def __init__(self, grid=8, arm="emergence", head="concat", t_per_frame=False,
                 n_layers=4):
        super().__init__()
        assert grid & (grid - 1) == 0 and grid >= 4
        assert n_layers >= 4
        self.grid = grid
        self.arm = arm
        self.head_type = head
        self.t_per_frame = bool(t_per_frame)
        self.n_layers = n_layers
        self.c1 = nn.Conv3d(3, CH, kernel_size=(3, 3, 3))
        self.c2 = nn.Conv3d(CH, CH, kernel_size=(3, 3, 3))
        self.c3 = nn.Conv3d(CH, CH, kernel_size=(3, 3, 3))
        self.c4 = nn.Conv3d(CH, CH, kernel_size=(3, 3, 3))
        # E10.3: optional extra layers (temporal receptive field +-n_layers);
        # empty for the legacy default so state_dict keys are unchanged
        self.extra = nn.ModuleList(
            nn.Conv3d(CH, CH, kernel_size=(3, 3, 3)) for _ in range(n_layers - 4))
        self.out = nn.Conv3d(CH, 1, kernel_size=(1, 1, 1))

    def denoise(self, xt, hist, t):
        """xt: (B,8,H,W) noised canvas; hist: (B,8,H,W) clean prefix;
        t: (B,) long (legacy scalar level) or, with t_per_frame, (B,8) long
        per-canvas-frame levels -> (u, u) logits (B,8,H,W) (second = corr-less
        dummy, kept for interface parity with DiffusionModel)."""
        B, _, H, W = xt.shape
        vol = torch.cat([hist, xt], dim=1).unsqueeze(1)      # (B,1,16,H,W)
        if t.dim() == 2:                                     # (B,8) per-frame
            a_f = alpha_t(t).unsqueeze(-1).unsqueeze(-1).expand(B, 8, H, W)
            s_f = sigma_t(t).unsqueeze(-1).unsqueeze(-1).expand(B, 8, H, W)
            one = torch.ones(B, 8, H, W, device=xt.device, dtype=a_f.dtype)
            zero = torch.zeros(B, 8, H, W, device=xt.device, dtype=a_f.dtype)
            a = torch.cat([one, a_f], dim=1).unsqueeze(1)    # (B,1,16,H,W)
            s = torch.cat([zero, s_f], dim=1).unsqueeze(1)
        else:
            a = alpha_t(t).view(B, 1, 1, 1, 1)
            s = sigma_t(t).view(B, 1, 1, 1, 1)
            a = a.expand(B, 1, 16, H, W)
            s = s.expand(B, 1, 16, H, W)
        h = torch.cat([vol, a, s], dim=1)                    # (B,3,16,H,W)
        for c in (self.c1, self.c2, self.c3, self.c4, *self.extra):
            h = _circ3d_spatial(h, 1)
            h = F.pad(h, (0, 0, 0, 0, 1, 1))                 # zero temporal pad
            h = F.relu(c(h))
        u = self.out(h).squeeze(1)                           # (B,16,H,W)
        u = u[:, 8:]                                         # canvas frames
        return u, u

    def training_loss(self, frames_pm1, loss_mask=None, pos_weight=None):
        """frames_pm1: (B,1,16,H,W) ±1 -> (loss, teacher-forced pixel acc).
        With t_per_frame (E10 arm U), each canvas frame draws an independent
        stratified level (the schedule cube); legacy draws one shared level."""
        x0 = frames_pm1[:, 0, 8:]                            # (B,8,H,W) clean ±1
        B = x0.shape[0]
        if self.t_per_frame:
            t = stratified_t(B * 8, device=x0.device).view(B, 8)
            a = alpha_t(t).unsqueeze(-1).unsqueeze(-1)
            s = sigma_t(t).unsqueeze(-1).unsqueeze(-1)
        else:
            t = stratified_t(B, device=x0.device)
            a = alpha_t(t).view(B, 1, 1, 1)
            s = sigma_t(t).view(B, 1, 1, 1)
        xt = a * x0 + s * torch.randn_like(x0)
        u, _ = self.denoise(xt, frames_pm1[:, 0, :8], t)
        target = (x0 + 1) / 2
        if loss_mask is not None:
            bce = F.binary_cross_entropy_with_logits(u, target, reduction="none")
            keep = ~loss_mask
            loss = (bce * keep).sum() / keep.sum().clamp(min=1)
        else:
            pw = (torch.as_tensor(pos_weight, dtype=u.dtype, device=u.device)
                  if pos_weight is not None else None)
            loss = F.binary_cross_entropy_with_logits(
                u, target, pos_weight=pw)
        acc = ((u > 0) == (x0 > 0)).float().mean()
        return loss, acc

    @torch.no_grad()
    def rollout(self, prefix_pm1, nfe=T_STEPS - 1, ablate_canvas=False,
                schedule="uniform", commit=False, pending="clean"):
        """prefix_pm1: (B,1,8,H,W) ±1 -> predictions (B,8,H,W) uint8.
        Identical reverse chain to DiffusionModel.

        schedule (t_per_frame only; E10): "uniform" = all canvas frames share
        each chain level (the diagonal path); "frame_ar" = frame k is fully
        denoised before k+1 starts (the staircase path). commit (frame_ar
        only) hard-commits each finished frame as ±1 and takes each frame's
        prediction at its own final level -- the eval-only sampler swap.

        pending (frame_ar only; E10.5, SETTINGS S13): the level declared for
        frames that have not started yet (they hold the initial N(0,1) draw).
        "clean" = level 0, the legacy E10/D0 convention (harmless for the
        backward-only structured encoder, but an out-of-distribution input for
        this bidirectional denoiser: pure noise declared clean); "noise" =
        the chain's top level, the Diffusion-Forcing convention and the
        in-distribution input under per-frame-level training."""
        assert pending in ("clean", "noise")
        B, _, _, H, W = prefix_pm1.shape
        hist = prefix_pm1[:, 0]
        dev = prefix_pm1.device
        x = torch.randn(B, 8, H, W, device=dev)
        levels = chain_levels(nfe)
        if self.t_per_frame and schedule == "frame_ar":
            u = None
            preds = []
            for k in range(8):
                for i, t in enumerate(levels):
                    tcol = torch.zeros(B, 8, dtype=torch.long, device=dev)
                    tcol[:, k] = t
                    if pending == "noise":
                        tcol[:, k + 1:] = levels[0]
                    u, _ = self.denoise(x, hist, tcol)
                    if i == len(levels) - 1:
                        break
                    x0h = torch.tanh(u / 2)
                    a_cur, s_cur = alpha_t(t), sigma_t(t)
                    t_next = levels[i + 1]
                    eps_k = (x[:, k] - a_cur * x0h[:, k]) / s_cur
                    x = x.clone()
                    if t_next == t - 1:
                        a_prev, s_prev = alpha_t(t - 1), sigma_t(t - 1)
                        mu = (a_prev * SIGMA1 ** 2 / s_cur ** 2) * x0h[:, k] \
                            + (a_cur * s_prev ** 2 / s_cur ** 2) * x[:, k]
                        var = SIGMA1 ** 2 * s_prev ** 2 / s_cur ** 2
                        x[:, k] = mu + math.sqrt(var) * torch.randn_like(x[:, k])
                    else:
                        x[:, k] = alpha_t(t_next) * x0h[:, k] + sigma_t(t_next) * eps_k
                if commit:
                    pred_k = (u[:, k] > 0)
                    preds.append(pred_k.to(torch.uint8))
                    x = x.clone()
                    x[:, k] = pred_k.float() * 2 - 1
            if commit:
                return torch.stack(preds, dim=1)
            return (u > 0).to(torch.uint8)
        u = None
        for i, t in enumerate(levels):
            if ablate_canvas:
                x = torch.randn_like(x)
            if self.t_per_frame:
                tcol = torch.full((B, 8), t, dtype=torch.long, device=dev)
                u, _ = self.denoise(x, hist, tcol)
            else:
                u, _ = self.denoise(x, hist, torch.full((B,), t, dtype=torch.long,
                                                        device=dev))
            if ablate_canvas or i == len(levels) - 1:
                continue
            x0h = torch.tanh(u / 2)
            a_cur, s_cur = alpha_t(t), sigma_t(t)
            eps_hat = (x - a_cur * x0h) / s_cur
            t_next = levels[i + 1]
            if t_next == t - 1:
                a_prev, s_prev = alpha_t(t - 1), sigma_t(t - 1)
                mu = (a_prev * SIGMA1 ** 2 / s_cur ** 2) * x0h \
                    + (a_cur * s_prev ** 2 / s_cur ** 2) * x
                var = SIGMA1 ** 2 * s_prev ** 2 / s_cur ** 2
                x = mu + math.sqrt(var) * torch.randn_like(x)
            else:
                x = alpha_t(t_next) * x0h + sigma_t(t_next) * eps_hat
        return (u > 0).to(torch.uint8)

    @torch.no_grad()
    def rollout_window(self, prefix_pm1, window=1, polish=0, inject=None,
                       nfe=T_STEPS - 1, pending="clean"):
        """Sliding-window sampler (E10.4, t_per_frame only; S12 amendment).

        Frame k starts its reverse chain when frame k-window commits
        (start_k = start_{k-window} + len(levels), start_k = 0 for
        k < window); each active frame advances one chain level per global
        step and hard-commits at its own final level. window=1 reproduces
        frame_ar+commit; window>=8 starts all frames together (the diagonal).
        After all frames commit: optional `inject` (frame k, (y, x)) flips one
        committed cell; optional `polish` runs that many joint steps at levels
        polish..1 over the full canvas (no commit during polish).
        pending: level declared for not-yet-started frames, as in rollout().
        Returns (B,8,H,W) uint8 -- the post-polish canvas if polish else the
        committed predictions."""
        assert pending in ("clean", "noise")
        B, _, _, H, W = prefix_pm1.shape
        hist = prefix_pm1[:, 0]
        dev = prefix_pm1.device
        x = torch.randn(B, 8, H, W, device=dev)
        levels = chain_levels(nfe)
        S = len(levels)
        starts = [0 if k < window else (k - window + 1) * 0 for k in range(8)]
        for k in range(window, 8):
            starts[k] = starts[k - window] + S
        preds = torch.zeros(B, 8, H, W, dtype=torch.uint8, device=dev)
        done = [False] * 8
        u = None
        for g in range(starts[7] + S):
            active = [k for k in range(8)
                      if starts[k] <= g < starts[k] + S and not done[k]]
            if not active:
                continue
            tcol = torch.zeros(B, 8, dtype=torch.long, device=dev)
            for k in active:
                tcol[:, k] = levels[g - starts[k]]
            if pending == "noise":
                for k in range(8):
                    if g < starts[k]:
                        tcol[:, k] = levels[0]
            u, _ = self.denoise(x, hist, tcol)
            x = x.clone()
            for k in active:
                i = g - starts[k]
                t = levels[i]
                if i == S - 1:
                    pk = (u[:, k] > 0)
                    preds[:, k] = pk.to(torch.uint8)
                    x[:, k] = pk.float() * 2 - 1
                    done[k] = True
                    continue
                x0h = torch.tanh(u / 2)
                a_cur, s_cur = alpha_t(t), sigma_t(t)
                t_next = levels[i + 1]
                if t_next == t - 1:
                    a_prev = alpha_t(t - 1)
                    s_prev = sigma_t(t - 1)
                    mu = (a_prev * SIGMA1 ** 2 / s_cur ** 2) * x0h[:, k] \
                        + (a_cur * s_prev ** 2 / s_cur ** 2) * x[:, k]
                    var = SIGMA1 ** 2 * s_prev ** 2 / s_cur ** 2
                    x[:, k] = mu + math.sqrt(var) * torch.randn_like(x[:, k])
                else:
                    eps_k = (x[:, k] - a_cur * x0h[:, k]) / s_cur
                    x[:, k] = alpha_t(t_next) * x0h[:, k] \
                        + sigma_t(t_next) * eps_k
        if inject is not None:
            k, (y, xx) = inject
            x[:, k, y, xx] = -x[:, k, y, xx]
        if polish > 0:
            for t in range(polish, 0, -1):
                tcol = torch.full((B, 8), t, dtype=torch.long, device=dev)
                u, _ = self.denoise(x, hist, tcol)
                if t == 1:
                    break
                x0h = torch.tanh(u / 2)
                a_cur, s_cur = alpha_t(t), sigma_t(t)
                a_prev, s_prev = alpha_t(t - 1), sigma_t(t - 1)
                mu = (a_prev * SIGMA1 ** 2 / s_cur ** 2) * x0h \
                    + (a_cur * s_prev ** 2 / s_cur ** 2) * x
                var = SIGMA1 ** 2 * s_prev ** 2 / s_cur ** 2
                x = mu + math.sqrt(var) * torch.randn_like(x)
            return (u > 0).to(torch.uint8)
        return preds

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
