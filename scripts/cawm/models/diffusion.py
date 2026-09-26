"""Diffusion world model (momentum-consistent redesign), docs/SETTINGS.md §6.5.

Frozen spec (2026-08-29 chat rounds):

  Forward: VP-AR(1), x_{t+1} = 0.9·x_t + √0.19·ε. Closed form
  x_t = 0.9^t·x₀ + √(1−0.81^t)·ε with Var(x_t) ≡ 1 and stationary N(0,1);
  T = 50 levels (a₄₄ < 0.01). Training noises the 8 output frames jointly
  (3D); history frames stay clean (inpaint) so the condition is t-independent.

  Training: single-step denoising, no chain. Per sample t ~ U{0..49} with
  batch stratification; denoiser emits per-pixel raw logits u; loss
  BCE-with-logits(u, (x₀+1)/2) — no tanh, no clamp (|dL/du| <= 1 automatic).

  Inference: recurrent reverse chain along the noise axis only (frames joint).
  Per level t: u = Net(x_t, t, cond) -> x̂₀ = tanh(u/2) in (−1,1) (bounded
  belief E[x₀]); consecutive levels take the analytic posterior
    x_{t−1} = (a^{t−1}σ²/σ_t²)·x̂₀ + (a·σ_{t−1}²/σ_t²)·x_t + √(σ²σ_{t−1}²/σ_t²)·ζ
  (Monte-Carlo verified); sub-sampled NFE jumps use the DDIM deterministic
  step. Final frame = sign(u).

  Condition: the constructive A/B stacks (§6) — 36-d rule-evidence condition
  from the clean prefix + per-pixel 18-d (s,n) codes. v2 head (registered
  2026-08-29 after the batch-3 diagnosis): a belief pass first estimates the
  clean canvas from noisy evidence; the (s,n) codes for the lookup are
  computed from that BELIEF (frame 0 uses the clean hist[-1] directly), so
  the codes-only lookup pathway stays near-one-hot at every noise level.
  Final logit = lookup(belief-codes) + corr(noisy evidence). The isolated
  probe showed a codes+cond MLP reaches 1.0000 on the clean one-step task in
  <=500 steps, while codes computed on noisy frames cap the shared head at
  the ~0.86 marginal prior (the batch-3 v1 failure mode).

  Canvas encoder (trained in both arms): two Conv3d(·->FEAT_CH,(2,3,3)) +
  ReLU with circular spatial padding and a zero temporal pad on the right.

  E3.7 "true diffusion" flags (2026-08-31, all default-off = legacy):
  corr_rf="bwd" (frame k sees canvas frames k-1,k — CA-causal direction),
  prepend_hist (clean hist[-1] prepended as canvas frame 0: never noised,
  (a,sigma)=(1,0), excluded from the output — true inpainting anchor),
  spin_scale S (targets ±S instead of ±1; inference belief x0h = S*tanh(u/2);
  the BCE loss is scale-free and unchanged), canvas_tanh (encoder reads
  tanh(canvas) — the black/white belief, consistent between training and
  inference), corr_head="bilinear" (canvas-feature x rule-condition
  cross-gating: out(relu(W_f([feat|a,s]) * W_c(cond))), mirrors the lookup
  bilinear head), and t_frame_slope_now (per-frame training-noise offset
  slope*k, set per-step by train.py; 0 = off)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constructive_cnn import (
    SumPool2,
    _build_analytic_a,
    _build_analytic_b,
    _circ2d,
    _circ3d,
    apply_agg_init,
)

T_STEPS = 50        # noise levels t = 0..49; a_44 < 0.01 (SNR -40 dB)
RETAIN = 0.9        # per-step signal retention (author-specified 0.9)
SIGMA1 = math.sqrt(1.0 - RETAIN ** 2)   # per-step noise std = 0.43589
FEAT_CH = 32        # canvas encoder width
CODE_HIDDEN = 64    # lookup-pathway width ([codes | cond] -> logit)
CORR_HIDDEN = 64    # canvas-evidence correction pathway width
HEAD_HIDDEN = None  # deprecated (kept for arg compat)


def alpha_t(t):
    """Signal retention at level t: a_t = 0.9^t (scalar or tensor)."""
    return RETAIN ** t


def sigma_t(t):
    """Cumulative noise std at level t: sqrt(1 - 0.81^t)."""
    return torch.sqrt(1.0 - RETAIN ** (2 * t)) if torch.is_tensor(t) \
        else math.sqrt(max(0.0, 1.0 - RETAIN ** (2 * t)))


def stratified_t(batch, device="cpu"):
    """Batch-stratified noise levels: each of the T levels appears
    floor(batch/T) times, the remainder spread by randperm, then shuffled.
    Consumes the global torch RNG (documented determinism: seeded once in
    train.py; model init is isolated from this stream)."""
    q, r = divmod(batch, T_STEPS)
    parts = [torch.arange(T_STEPS).repeat(q)]
    if r:
        parts.append(torch.randperm(T_STEPS)[:r])
    t = torch.cat(parts)
    return t[torch.randperm(batch)].to(device)


def chain_levels(nfe):
    """Descending noise levels for the reverse chain (all in 1..49).

    nfe >= 49 -> the full ancestral chain (levels 49..1, all consecutive
    steps use the frozen stochastic posterior). nfe < 49 -> uniform
    sub-sample including both endpoints; jumps use the DDIM step."""
    if nfe >= T_STEPS - 1:
        return list(range(T_STEPS - 1, 0, -1))
    idx = torch.linspace(0, T_STEPS - 2, nfe).round().long().unique()
    return sorted((T_STEPS - 1 - idx).tolist(), reverse=True)


class DiffusionModel(nn.Module):
    def __init__(self, grid=8, arm="existence", cascade_feedback="tanh",
                 lookup_head="concat", corr_rf="fwd", t_per_frame=False,
                 no_cascade=False, corr_head="mlp", prepend_hist=False,
                 spin_scale=1.0, canvas_tanh=False,
                 chain_nfe=0, chain_init="noise", frontier=False,
                 det_feat=3, det_split=27, agg_init="ones", cond_pairing="conv"):
        super().__init__()
        assert arm in ("existence", "emergence")
        assert cascade_feedback in ("tanh", "sign")
        assert lookup_head in ("concat", "bilinear")
        assert corr_rf in ("fwd", "sym", "bwd")
        assert corr_head in ("mlp", "bilinear")
        assert grid & (grid - 1) == 0 and grid >= 4
        assert cond_pairing in ("conv", "kvshift")
        self.cond_pairing = cond_pairing
        self.grid = grid
        self.arm = arm
        self.cascade_feedback = cascade_feedback
        self.lookup_head = lookup_head
        self.corr_rf = corr_rf
        self.corr_head = corr_head
        self.prepend_hist = bool(prepend_hist)
        self.spin_scale = float(spin_scale)
        self.canvas_tanh = bool(canvas_tanh)
        self.t_per_frame = bool(t_per_frame)
        self.no_cascade = bool(no_cascade)
        self.t_frame_slope_now = 0.0   # set per-step by train.py (E3.7 arm)
        # E3.8 chain config: when chain_nfe > 0, BOTH training (chain_loss)
        # and inference (rollout) run the sub-sampled reverse chain of that
        # many levels, starting from chain_init ("zeros" = neutral belief
        # canvas, protocol note 2026-09-01; "noise" = N(0,1) legacy start).
        # 0/None = legacy: full 49-level chain from N(0,1), single-step
        # marginal training (default off = legacy bitwise).
        assert chain_init in ("zeros", "noise")
        self.chain_nfe = int(chain_nfe) if chain_nfe else None
        self.chain_init = chain_init
        # E3.9 frontier training (author framework 2026-09-02): when set,
        # training uses frontier_loss (teacher-forced clean prefix + noised
        # frontier frame + loss on the frontier frame ONLY) and rollout
        # auto-routes to the matched frame_ar+commit schedule.
        self.frontier = bool(frontier)
        # E6.6g detector redundancy (emergence only; 3/27 = registered, bitwise)
        self.det_feat, self.det_split = int(det_feat), int(det_split)
        # E4.6 aggregation init control (emergence only; 'ones' = registered)
        self.agg_init = agg_init

        # ---- condition: constructive A stack ----
        if arm == "existence":
            self.feat_a = nn.Conv3d(1, 3, kernel_size=(2, 3, 3))
            self.split_a = nn.Conv3d(3, 24, 1)
            self.conjoin_a = nn.Conv3d(24, 36, 1)
            _build_analytic_a(self.feat_a, self.split_a, self.conjoin_a)
            for m in (self.feat_a, self.split_a, self.conjoin_a):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        else:
            self.feat_a = nn.Conv3d(1, self.det_feat, kernel_size=(2, 3, 3))
            self.split_a = nn.Conv3d(self.det_feat, self.det_split, 1)
            self.conjoin_a = nn.Conv3d(self.det_split, 36, 1)
            self.gate_theta = nn.Parameter(torch.zeros(36))
            n_levels = int(torch.log2(torch.tensor(float(grid))).item()) - 1
            self.pyramid = nn.ModuleList(
                [nn.Conv3d(36, 36, kernel_size=(1, 3, 3), groups=36)
                 for _ in range(n_levels)]
            )
            for conv in self.pyramid:
                nn.init.ones_(conv.weight)
                nn.init.zeros_(conv.bias)
            self.pool = SumPool2()
            self.temporal_w = nn.Parameter(torch.ones(36, 7))
            self.temporal_b = nn.Parameter(torch.zeros(36))

        # ---- condition: constructive B stack ----
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

        # ---- canvas encoder (trained in both arms) ----
        # corr_rf="fwd": kernel (2,3,3), right temporal zero pad -> frame k
        #   sees canvas frames k..k+1 (forward-only receptive field).
        # corr_rf="sym": kernel (3,3,3), BOTH-sided temporal zero pad (1 each
        #   side) -> frame k sees canvas frames k-1..k+1 (symmetric RF).
        # corr_rf="bwd": kernel (2,3,3), LEFT-only temporal zero pad ->
        #   frame k sees canvas frames k-1..k (CA-causal direction, E3.7).
        t_kernel = 2 if corr_rf in ("fwd", "bwd") else 3
        self.enc1 = nn.Conv3d(1, FEAT_CH, kernel_size=(t_kernel, 3, 3))
        self.enc2 = nn.Conv3d(FEAT_CH, FEAT_CH, kernel_size=(t_kernel, 3, 3))

        # ---- v3 head (iterative-lookup cascade, registered 2026-08-29 late):
        # The board is generated by a rule CHAIN; the denoiser mirrors that
        # with an explicit 8-step inner recursion:
        #   u_0 = lookup([B(hist[-1]) | cond])            (clean codes, exact)
        #   u_k = lookup([B(tanh(u_{k-1}/2)) | cond]) + corr(feat_k, cond, (a,s))
        # v2's belief pass could not crystallize the board (no per-cell clean
        # evidence; corr drowned the exact frame-0 lookup: frame-0 acc at t=49
        # was 0.84). v3 puts the recursion in the lookup itself: frame 0 is
        # exact at EVERY level, and correctness propagates frame by frame.
        # lookup takes cond too (rule identity) so it generalizes beyond L1.
        if lookup_head == "concat":
            self.lookup = nn.Sequential(
                nn.Linear(18 + 36, CODE_HIDDEN), nn.ReLU(), nn.Linear(CODE_HIDDEN, 1)
            )
        else:  # bilinear: mirrors ConstructiveCNN's bilinear head (§6 Head-A)
            self.W_c = nn.Linear(18, 36)
            self.W_e = nn.Linear(36, 36)
            self.out = nn.Linear(36, 1)
        # corr heads are mutually exclusive so each variant keeps its own
        # parameter count (and legacy checkpoints stay loadable).
        if corr_head == "mlp":
            c_evid = FEAT_CH + 38
            self.corr = nn.Sequential(
                nn.Conv2d(c_evid, CORR_HIDDEN, 1), nn.ReLU(),
                nn.Conv2d(CORR_HIDDEN, 1, 1),
            )
        else:
            # bilinear corr head (E3.7): out(relu(W_f([feat|a,s]) * W_c(cond)))
            # — the rule condition cross-gates the canvas feature (linear
            # cross-attention), instead of letting a concat MLP learn the
            # interaction from data. Param-matched to the mlp (4673 vs 4609).
            self.corr_wf = nn.Conv2d(FEAT_CH + 2, CORR_HIDDEN, 1)
            self.corr_wc = nn.Conv2d(36, CORR_HIDDEN, 1)
            self.corr_out = nn.Conv2d(CORR_HIDDEN, 1, 1)

        if arm == "emergence":
            apply_agg_init(self.pyramid, self.temporal_w, self.temporal_b, agg_init)

        # E12.5: construct last so all common modules retain legacy init.
        if self.cond_pairing == "kvshift":
            del self.feat_a, self.split_a, self.conjoin_a
            self.cond_feat = nn.Conv2d(1, 2, 3)
            self.cond_split = nn.Conv2d(2, 22 if arm == "existence" else 18, 1)
            self.cond_conjoin = nn.Conv2d(22 if arm == "existence" else 18, 18, 1)
            if arm == "existence":
                _build_analytic_b(self.cond_feat, self.cond_split, self.cond_conjoin)
                for m in (self.cond_feat, self.cond_split, self.cond_conjoin):
                    for p in m.parameters(): p.requires_grad_(False)

    # ---- condition ----
    def forward_a(self, prefix_pm1):
        """prefix_pm1: (B,1,8,H,W) ±1 -> condition (B,36). Mirrors §6."""
        if self.cond_pairing == "conv":
            h = self.feat_a(_circ3d(prefix_pm1, 1))
            h = F.relu(self.split_a(h))
            h = F.relu(self.conjoin_a(h))
        else:
            B, _, T, H, W = prefix_pm1.shape
            frames = prefix_pm1[:, 0]
            z = frames[:, :-1].reshape(B * (T-1), 1, H, W)
            z = self.cond_feat(_circ2d(z, 1))
            z = F.relu(self.cond_split(z))
            z = F.relu(self.cond_conjoin(z)).reshape(B,T-1,18,H,W)
            outcome = ((frames[:, 1:] + 1) / 2).unsqueeze(2)
            h = torch.stack((z * (1-outcome), z * outcome),dim=3)
            h = h.reshape(B,T-1,36,H,W).permute(0,2,1,3,4)
        if self.arm == "existence":
            h = h.sum(dim=(3, 4))
            cond = h.sum(dim=-1)
        else:
            h = F.relu(h - self.gate_theta[None, :, None, None, None])
            for conv in self.pyramid:
                h = self.pool(conv(_circ3d(h, 1)))
            h = h.sum(dim=(3, 4))
            cond = torch.einsum("bct,ct->bc", h, self.temporal_w) + self.temporal_b
        return torch.log1p(cond.clamp(min=0))

    def forward_b(self, frames_pm1):
        """frames_pm1: (N,1,H,W) ±1 -> codes (N,18,H,W)."""
        h = self.feat_b(_circ2d(frames_pm1, 1))
        h = F.relu(self.split_b(h))
        return F.relu(self.conjoin_b(h))

    # ---- denoiser ----
    def _encode(self, canvas):
        """canvas: (B,1,F,H,W) -> features (B,FEAT_CH,F,H,W)."""
        if self.corr_rf == "fwd":
            tpad = (0, 1)          # right-only temporal zero pad
        elif self.corr_rf == "bwd":
            tpad = (1, 0)          # left-only: frame k sees frames k-1, k
        else:                      # sym: one frame each side
            tpad = (1, 1)
        h = F.pad(canvas, (1, 1, 1, 1, 0, 0), mode="circular")
        h = F.pad(h, (0, 0, 0, 0, tpad[0], tpad[1]))
        h = F.relu(self.enc1(h))
        h = F.pad(h, (1, 1, 1, 1, 0, 0), mode="circular")
        h = F.pad(h, (0, 0, 0, 0, tpad[0], tpad[1]))
        return F.relu(self.enc2(h))

    def _lookup_logits(self, codes, cond_hw):
        """codes: (B,18,H,W); cond_hw: (B,36,H,W) -> per-cell logits (B,H,W).
        concat head: MLP([codes|cond]); bilinear head: out(relu(W_c(codes) *
        W_e(cond))) per cell (mirrors ConstructiveCNN's Head-A)."""
        B, _, H, W = codes.shape
        if self.lookup_head == "concat":
            z = torch.cat([codes, cond_hw], dim=1)         # (B,54,H,W)
            uk = self.lookup(z.permute(0, 2, 3, 1).reshape(-1, 54))
            return uk.reshape(B, H, W)
        c = self.W_c(codes.permute(0, 2, 3, 1))            # (B,H,W,36)
        e = self.W_e(cond_hw.permute(0, 2, 3, 1))          # (B,H,W,36)
        return self.out(F.relu(c * e)).squeeze(-1)         # (B,H,W)

    def denoise(self, xt, hist, t):
        """xt: (B,8,H,W) noised canvas; hist: (B,8,H,W) clean prefix frames;
        t: (B,) long levels (shared) or (B,8) per-frame levels when
        t_per_frame -> (u, u_corr) logits (B,8,H,W).

        v3 iterative-lookup cascade (see __init__ comment): frame 0 from the
        clean hist[-1] codes, frame k from the model's own frame k-1 answer
        through the frozen/learned B stack; the canvas-evidence corr pathway
        adds level-dependent residual evidence. no_cascade skips the lookup
        loop entirely (u = u_corr)."""
        B, _, H, W = xt.shape
        cond = self.forward_a(hist.unsqueeze(1))           # (B,36)
        # per-frame (a, sigma) condition for the 8 OUTPUT frames
        if self.t_per_frame:
            ats = torch.stack([alpha_t(t), sigma_t(t)], dim=1)   # (B,2,8)
        else:
            ats = torch.stack([alpha_t(t), sigma_t(t)], dim=1)   # (B,2)
            ats = ats[:, :, None].expand(B, 2, 8)
        # canvas: optionally prepend the clean hist[-1] (never noised,
        # (a,sigma)=(1,0)) as frame 0 — the true-inpainting anchor (E3.7)
        if self.prepend_hist:
            canvas = torch.cat([self.spin_scale * hist[:, -1:], xt], dim=1)
            a0 = torch.ones(B, 2, 1, device=xt.device, dtype=ats.dtype)
            a0[:, 1, 0] = 0.0
            ats = torch.cat([a0, ats], dim=2)                    # (B,2,9)
        else:
            canvas = xt
        Fc = canvas.shape[1]
        e = torch.cat([cond[:, :, None].expand(B, 36, Fc), ats], dim=1)
        e = e[:, :, :, None, None].expand(B, 38, Fc, H, W)
        enc_in = torch.tanh(canvas) if self.canvas_tanh else canvas
        feat = self._encode(enc_in.unsqueeze(1))           # (B,FEAT_CH,Fc,H,W)
        if self.corr_head == "mlp":
            ev = torch.cat([feat, e], dim=1).permute(0, 2, 1, 3, 4)
            ev = ev.reshape(B * Fc, -1, H, W)
            u_corr = self.corr(ev).reshape(B, Fc, H, W)
        else:
            f_part = torch.cat([feat, e[:, 36:]], dim=1)   # (B,F+2,Fc,H,W)
            f_part = f_part.permute(0, 2, 1, 3, 4).reshape(B * Fc, -1, H, W)
            c_part = e[:, :36].permute(0, 2, 1, 3, 4).reshape(B * Fc, 36, H, W)
            u_corr = self.corr_out(
                F.relu(self.corr_wf(f_part) * self.corr_wc(c_part))
            ).reshape(B, Fc, H, W)
        if self.prepend_hist:
            u_corr = u_corr[:, 1:]                             # (B,8,H,W)

        if self.no_cascade:
            return u_corr, u_corr

        cond_hw = cond[:, :, None, None].expand(B, 36, H, W)
        prev = hist[:, -1]                                 # (B,H,W) clean
        us = []
        for k in range(8):
            codes = self.forward_b(prev.unsqueeze(1))      # (B,18,H,W)
            uk = self._lookup_logits(codes, cond_hw)
            us.append(uk)
            # detached soft-belief feedback: each frame's lookup is a clean
            # supervised task on the model's own current answers; cutting the
            # cross-frame gradient prevents corr-vs-cascade co-adaptation
            # (v3 joint-gradient plateaued at ~0.89; v3.1 detaches).
            if self.cascade_feedback == "sign":
                prev = (uk.detach() > 0).float() * 2 - 1
            else:
                prev = torch.tanh(uk.detach() / 2)
        u_look = torch.stack(us, dim=1)                    # (B,8,H,W)
        return u_look + u_corr, u_corr

    # ---- training (single step, no chain) ----
    def training_loss(self, frames_pm1, loss_mask=None):
        """frames_pm1: (B,1,16,H,W) ±1 -> (loss, teacher-forced pixel acc).
        loss_mask: optional (B,8,H,W) bool; True marks UNPREDICTABLE cells
        excluded from the BCE (masked mean over predictable cells only)."""
        x0 = frames_pm1[:, 0, 8:]                          # (B,8,H,W) clean ±1
        B = x0.shape[0]
        if self.t_per_frame:
            t = stratified_t(B * 8, device=x0.device).reshape(B, 8)
            slope = float(getattr(self, "t_frame_slope_now", 0.0))
            if slope > 0.0:
                # E3.7 curriculum: frame k noised at t + slope*k (earlier
                # frames cleaner); annealed to 0 by train.py (legacy: 0).
                off = slope * torch.arange(8, device=x0.device,
                                           dtype=torch.float32)
                t = (t.float() + off[None, :]).clamp(max=T_STEPS - 1).long()
            a = alpha_t(t).view(B, 8, 1, 1)
            s = sigma_t(t).view(B, 8, 1, 1)
        else:
            t = stratified_t(B, device=x0.device)
            a = alpha_t(t).view(B, 1, 1, 1)
            s = sigma_t(t).view(B, 1, 1, 1)
        xt = a * (self.spin_scale * x0) + s * torch.randn_like(x0)
        u, _ = self.denoise(xt, frames_pm1[:, 0, :8], t)
        target = (x0 + 1) / 2
        if loss_mask is not None:
            bce = F.binary_cross_entropy_with_logits(u, target, reduction="none")
            keep = ~loss_mask
            loss = (bce * keep).sum() / keep.sum().clamp(min=1)
        else:
            loss = F.binary_cross_entropy_with_logits(u, target)
        acc = ((u > 0) == (x0 > 0)).float().mean()
        return loss, acc

    # ---- E3.9: frontier training (matched to frame_ar+commit inference) ----
    def frontier_loss(self, frames_pm1, evidence_groups=None, evidence_mode="none"):
        """Per-level supervision by COMMITMENT SCHEDULE, not by weighting:
        draw a frontier frame k ~ U{0..7} per sample; canvas = exactly-clean
        teacher-forced prefix (spin_scale * gt, tcol=0) + frontier frame
        noised at a stratified level t + pure-noise future frames (tcol=T-1,
        matching frame_ar inference start); BCE on the frontier frame ONLY.
        Each level's optimum is "extend the committed prefix by one frame" —
        the per-level tasks differ, so this is the non-redundant form of
        per-level supervision (supervising beyond-frontier frames toward x0
        would recreate the same-target collapse seen in E3.8)."""
        x0 = frames_pm1[:, 0, 8:]                          # (B,8,H,W) clean ±1
        hist = frames_pm1[:, 0, :8]
        B, _, H, W = x0.shape
        dev = x0.device
        k = torch.randint(0, 8, (B,), device=dev)          # frontier frame
        t = stratified_t(B, device=dev)                    # frontier level
        a = alpha_t(t).view(B, 1, 1)
        s = sigma_t(t).view(B, 1, 1)
        idx = torch.arange(8, device=dev)[None, :]         # (1,8)
        committed = idx < k[:, None]                       # (B,8)
        xt = torch.randn(B, 8, H, W, device=dev)           # future: N(0,1)
        xt = torch.where(committed[:, :, None, None],
                         self.spin_scale * x0, xt)         # prefix: clean gt
        xk = x0[torch.arange(B, device=dev), k]            # (B,H,W)
        xt[torch.arange(B, device=dev), k] = \
            a * (self.spin_scale * xk) + s * torch.randn(B, H, W, device=dev)
        tcol = torch.full((B, 8), T_STEPS - 1, dtype=torch.long, device=dev)
        tcol[committed] = 0
        tcol[torch.arange(B, device=dev), k] = t
        u, _ = self.denoise(xt, hist, tcol)
        uk = u[torch.arange(B, device=dev), k]             # (B,H,W)
        target = (xk + 1) / 2
        if evidence_groups is None:
            loss = F.binary_cross_entropy_with_logits(uk, target)
        else:
            from ..evidence_loss import evidence_bce
            selected = evidence_groups[torch.arange(B, device=dev), k]
            loss = evidence_bce(uk, target, selected, evidence_mode)
        acc = ((uk > 0) == (xk > 0)).float().mean()
        return loss, acc

    # ---- E3.8: chain training (differentiable reverse chain) ----
    def chain_loss(self, frames_pm1, aux_weight=0.2):
        """Differentiable `chain_nfe`-step reverse chain; loss on the FINAL
        level's logits plus aux_weight * per-step BCE at every intermediate
        level. Replaces training_loss when chain_nfe is set.

        Why (E3.8 preregistration): (a) the chain canvas never contains the
        ground truth — it holds the model's own belief (zeros start = no
        information at all), so the marginal read-the-leak shortcut is
        structurally unavailable; (b) the training canvas distribution equals
        the inference chain distribution (exposure bias closed); (c) the
        gradient reaches EARLY levels through the canvas pathway, so the loss
        can finally see and reward the commit-early-frames allocation that
        the per-(t,frame) marginal loss is blind to.

        Sub-sampled levels (chain_levels) are non-consecutive, so every
        transition is the deterministic DDIM step — the whole chain is
        deterministic given the batch (zeros start: fully deterministic;
        noise start: one initial draw). SC stream: mask is a no-op (rate 0),
        so loss_mask is not taken."""
        x0 = frames_pm1[:, 0, 8:]                          # (B,8,H,W) clean ±1
        hist = frames_pm1[:, 0, :8]
        B, _, H, W = x0.shape
        dev = x0.device
        levels = chain_levels(self.chain_nfe)
        if self.chain_init == "zeros":
            x = torch.zeros(B, 8, H, W, device=dev)
        else:
            x = torch.randn(B, 8, H, W, device=dev)
        target = (x0 + 1) / 2
        losses = []
        u = None
        for i, t in enumerate(levels):
            if self.t_per_frame:
                tcol = torch.full((B, 8), t, dtype=torch.long, device=dev)
            else:
                tcol = torch.full((B,), t, dtype=torch.long, device=dev)
            u, _ = self.denoise(x, hist, tcol)
            losses.append(F.binary_cross_entropy_with_logits(u, target))
            if i == len(levels) - 1:
                break
            x0h = self.spin_scale * torch.tanh(u / 2)      # bounded belief
            a_cur, s_cur = alpha_t(t), sigma_t(t)
            eps_hat = (x - a_cur * x0h) / s_cur
            t_next = levels[i + 1]
            if t_next == t - 1:                            # frozen posterior
                a_prev, s_prev = alpha_t(t - 1), sigma_t(t - 1)
                mu = (a_prev * SIGMA1 ** 2 / s_cur ** 2) * x0h \
                    + (a_cur * s_prev ** 2 / s_cur ** 2) * x
                var = SIGMA1 ** 2 * s_prev ** 2 / s_cur ** 2
                x = mu + math.sqrt(var) * torch.randn_like(x)
            else:                                          # DDIM jump
                x = alpha_t(t_next) * x0h + sigma_t(t_next) * eps_hat
        loss = losses[-1]
        if aux_weight > 0 and len(losses) > 1:
            loss = loss + aux_weight * torch.stack(losses[:-1]).sum()
        acc = ((u > 0) == (x0 > 0)).float().mean()
        return loss, acc

    # ---- inference: recurrent reverse chain ----
    @torch.no_grad()
    def rollout(self, prefix_pm1, nfe=T_STEPS - 1, ablate_canvas=False,
                schedule="uniform", commit=False):
        """prefix_pm1: (B,1,8,H,W) ±1 -> predictions (B,8,H,W) uint8.

        ablate_canvas=True re-draws the chain canvas from N(0,1) at every
        level and skips the posterior update, so the corr pathway only ever
        sees pure noise: the gap between normal and ablated SeqAcc measures
        how much the model actually uses the canvas (0 for a fully
        cascade-driven model; the joint-generation hypothesis predicts the
        gap grows with training).

        schedule (t_per_frame only): "uniform" (all frames share each level,
        the legacy path), "staggered" (frame k uses levels offset by
        k*T/8), "frame_ar" (frame k fully denoised before k+1 starts).
        With t_per_frame off, schedule is ignored (uniform legacy).

        commit (frame_ar only, default off = E3.2-bitwise): hard-commit
        each finished frame into the canvas as spin_scale*sign(u) — the
        committed prefix is then exactly clean, matching the tcol=0
        declaration — and take each frame's prediction at its own final
        level (D0 probe, 2026-09-02)."""
        if self.frontier and schedule == "uniform":
            # E3.9: a frontier-trained model evals with its matched schedule
            schedule, commit = "frame_ar", True
        B, _, _, H, W = prefix_pm1.shape
        hist = prefix_pm1[:, 0]                            # (B,8,H,W) clean
        dev = prefix_pm1.device
        # E3.8: a chain-configured model evals with ITS chain (sub-sampled
        # levels + configured start canvas); explicit nfe still overrides.
        levels = chain_levels(nfe if nfe != T_STEPS - 1
                              else (self.chain_nfe or nfe))
        if self.chain_init == "zeros" and nfe == T_STEPS - 1:
            x = torch.zeros(B, 8, H, W, device=dev)        # neutral belief
        else:
            x = torch.randn(B, 8, H, W, device=dev)        # N(0,1) legacy

        if self.t_per_frame and schedule == "frame_ar":
            # Frame-autoregressive: fully denoise frame k (cascade-free
            # logits only) before starting frame k+1.
            x = torch.randn(B, 8, H, W, device=dev)
            u = None
            preds = []
            for k in range(8):
                for i, t in enumerate(levels):
                    tcol = torch.zeros(B, 8, dtype=torch.long, device=dev)
                    tcol[:, k] = t
                    u, _ = self.denoise(x, hist, tcol)
                    if i == len(levels) - 1:
                        break
                    x0h = self.spin_scale * torch.tanh(u / 2)
                    a_cur, s_cur = alpha_t(t), sigma_t(t)
                    t_next = levels[i + 1]
                    # single-frame posterior/DDIM update on frame k only
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
                    x[:, k] = self.spin_scale * (pred_k.float() * 2 - 1)
            if commit:
                return torch.stack(preds, dim=1)
            return (u > 0).to(torch.uint8)

        x = torch.randn(B, 8, H, W, device=dev)            # N(0,1)
        u = None
        for i, t in enumerate(levels):
            if ablate_canvas:
                x = torch.randn_like(x)
            if self.t_per_frame:
                if schedule == "staggered":
                    # frame k uses levels offset by k*T/8 (clamped to >=1)
                    off = (torch.arange(8, device=dev) * (T_STEPS // 8))
                    tt = (t - off).clamp(min=1)
                    tcol = tt[None, :].expand(B, 8).contiguous()
                else:  # uniform: all frames share level t
                    tcol = torch.full((B, 8), t, dtype=torch.long, device=dev)
                u, _ = self.denoise(x, hist, tcol)
            else:
                u, _ = self.denoise(x, hist, torch.full((B,), t, dtype=torch.long,
                                                        device=dev))
            if ablate_canvas or i == len(levels) - 1:
                continue
            x0h = self.spin_scale * torch.tanh(u / 2)      # bounded belief
            a_cur, s_cur = alpha_t(t), sigma_t(t)
            eps_hat = (x - a_cur * x0h) / s_cur
            t_next = levels[i + 1]
            if t_next == t - 1:                            # frozen posterior
                a_prev, s_prev = alpha_t(t - 1), sigma_t(t - 1)
                mu = (a_prev * SIGMA1 ** 2 / s_cur ** 2) * x0h \
                    + (a_cur * s_prev ** 2 / s_cur ** 2) * x
                var = SIGMA1 ** 2 * s_prev ** 2 / s_cur ** 2
                x = mu + math.sqrt(var) * torch.randn_like(x)
            else:                                          # DDIM jump
                x = alpha_t(t_next) * x0h + sigma_t(t_next) * eps_hat
        return (u > 0).to(torch.uint8)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
