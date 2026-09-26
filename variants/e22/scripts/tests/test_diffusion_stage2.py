"""Diffusion stage-2 tests (§1.3): the four architecture fixes.

  cascade_feedback="sign"  — analytic existence arm + sign feedback rolls a
                             GoL blinker corpus exactly (SeqAcc 1.0 on l1_val_g8).
  lookup_head="bilinear"   — param count 2053; L1 overfit smoke (loss drops).
  corr_rf="sym"            — directional: perturbing canvas frame k-1 changes
                             the k-th encoder output slice (not with "fwd").
  t_per_frame / no_cascade — per-frame closed-form marginal matches Monte-Carlo;
                             shared-t + uniform schedule == legacy path bit-wise.
"""

import math

import numpy as np
import pytest
import torch

from cawm import rules as R
from cawm.data import StreamDataset, build_eval_corpus, load_eval_corpus
from cawm.eval import metrics, rollout_corpus
from cawm.models import DiffusionModel
from cawm.models.diffusion import (RETAIN, SIGMA1, T_STEPS, alpha_t, sigma_t,
                                   stratified_t)


# ---------------------------------------------------------- cascade_feedback

@pytest.mark.slow
def test_sign_feedback_blinker_exact(tmp_path):
    """Existence arm + sign feedback must roll a GoL corpus exactly."""
    path = str(tmp_path / "l1.npz")
    build_eval_corpus(path, n=128, master_seed=43, half="train",
                      rule_override=R.GOL_RULE)
    corpus = load_eval_corpus(path)
    torch.manual_seed(0)
    model = DiffusionModel(grid=8, arm="existence", cascade_feedback="sign")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=2e-3)
    stream = StreamDataset(42, half="train", length=400 * 64,
                           rule_override=R.GOL_RULE)
    model.train()
    for step in range(1, 401):
        idx = list(range((step - 1) * 64, step * 64))
        frames = stream.get_batch(idx)["frames"].float()
        loss, acc = model.training_loss((2 * frames - 1).unsqueeze(1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    preds = rollout_corpus(model, corpus)
    m = metrics(preds, corpus)
    assert m["seq_acc"] == 1.0, m


# ------------------------------------------------------------- lookup_head

def test_bilinear_lookup_param_count():
    m = DiffusionModel(grid=8, arm="existence", lookup_head="bilinear")
    # bilinear lookup = W_c 18*36+36 + W_e 36*36+36 + out 36+1 = 2053
    lookup_params = (m.W_c.weight.numel() + m.W_c.bias.numel() +
                     m.W_e.weight.numel() + m.W_e.bias.numel() +
                     m.out.weight.numel() + m.out.bias.numel())
    assert lookup_params == 2053


@pytest.mark.slow
def test_bilinear_lookup_overfit_smoke():
    torch.manual_seed(0)
    model = DiffusionModel(grid=8, arm="existence", lookup_head="bilinear")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=2e-3)
    stream = StreamDataset(42, half="train", length=200 * 64,
                           rule_override=R.GOL_RULE)
    model.train()
    losses = []
    for step in range(1, 201):
        idx = list(range((step - 1) * 64, step * 64))
        frames = stream.get_batch(idx)["frames"].float()
        loss, acc = model.training_loss((2 * frames - 1).unsqueeze(1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], (losses[0], losses[-1])


# ----------------------------------------------------------------- corr_rf

def test_corr_rf_sym_directional():
    """With corr_rf='sym', perturbing canvas frame k-1 changes the k-th output
    slice of the encoder; with 'fwd' it must not."""
    for rf, expect_change in (("sym", True), ("fwd", False)):
        torch.manual_seed(0)
        m = DiffusionModel(grid=8, arm="existence", corr_rf=rf).eval()
        xt = torch.randn(1, 1, 8, 8, 8)
        with torch.no_grad():
            base = m._encode(xt)
            xt2 = xt.clone()
            k = 4
            xt2[0, 0, k - 1] += 1.0          # perturb canvas frame k-1
            pert = m._encode(xt2)
        changed = not torch.allclose(base[0, :, k], pert[0, :, k])
        assert changed == expect_change, \
            f"rf={rf}: frame-{k} slice {'changed' if changed else 'unchanged'} " \
            f"by frame-{k-1} perturbation (expect change={expect_change})"


# ---------------------------------------------- t_per_frame / no_cascade

def test_per_frame_marginal_matches_monte_carlo():
    """Per-frame noising closed form: x_t = a_t*x0 + sigma_t*eps, checked per
    frame against the recurrent chain (mirror of the shared-t verification)."""
    rng = np.random.default_rng(0)
    N, t = 100_000, 7
    x0 = np.ones(N)
    x = x0.copy()
    for _ in range(t):
        x = RETAIN * x + SIGMA1 * rng.standard_normal(N)
    cf = RETAIN ** t * x0 + math.sqrt(1 - RETAIN ** (2 * t)) * rng.standard_normal(N)
    for xs in (x, cf):
        assert abs(xs.mean() - RETAIN ** t) < 0.02
        assert abs(xs.std() - math.sqrt(1 - RETAIN ** (2 * t))) < 0.02


def test_t_per_frame_stratified_shape():
    t = stratified_t(64 * 8).reshape(64, 8)
    assert t.shape == (64, 8)
    assert t.min() >= 0 and t.max() < T_STEPS


def test_shared_t_uniform_equals_legacy_bitwise():
    """With all frames sharing one t and schedule='uniform', the t_per_frame
    path must equal the legacy path bit-wise on a fixed seed."""
    torch.manual_seed(0)
    legacy = DiffusionModel(grid=8, arm="existence")
    torch.manual_seed(0)
    perframe = DiffusionModel(grid=8, arm="existence", t_per_frame=True)
    perframe.load_state_dict(legacy.state_dict())
    legacy.eval()
    perframe.eval()
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randn(2, 8, 8, 8).tanh()
    t_shared = torch.tensor([5, 9])
    with torch.no_grad():
        u_l, _ = legacy.denoise(xt, hist, t_shared)
        t_pf = t_shared[:, None].expand(2, 8).contiguous()
        u_p, _ = perframe.denoise(xt, hist, t_pf)
    assert torch.equal(u_l, u_p), "per-frame shared-t path diverges from legacy"


def test_no_cascade_returns_corr_only():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True).eval()
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randn(2, 8, 8, 8).tanh()
    with torch.no_grad():
        u, u_corr = m.denoise(xt, hist, torch.tensor([3, 7]))
    assert torch.equal(u, u_corr), "no_cascade must return u = u_corr"


# ------------------------------------------------------- E3.7 true diffusion

def test_corr_rf_bwd_directional():
    """With corr_rf='bwd', frame k sees canvas frames k-1 and k ONLY:
    perturbing canvas frame k-1 changes output slice k; perturbing canvas
    frame k+1 must NOT (the CA-causal direction)."""
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", corr_rf="bwd").eval()
    xt = torch.randn(1, 1, 8, 8, 8)
    k = 4
    with torch.no_grad():
        base = m._encode(xt)
        xt2 = xt.clone()
        xt2[0, 0, k - 1] += 1.0          # perturb PAST frame k-1
        past = m._encode(xt2)
        xt3 = xt.clone()
        xt3[0, 0, k + 1] += 1.0          # perturb FUTURE frame k+1
        fut = m._encode(xt3)
    assert not torch.allclose(base[0, :, k], past[0, :, k]), \
        "bwd: frame-k slice must respond to frame k-1"
    assert torch.allclose(base[0, :, k], fut[0, :, k]), \
        "bwd: frame-k slice must NOT respond to frame k+1"


def test_prepend_hist_anchor():
    """prepend_hist: output stays 8 frames, no new parameters, and the clean
    prepended frame anchors frame-0 prediction (flipping hist[-1] must change
    the frame-0 corr logits when bwd reads it)."""
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       corr_rf="bwd", t_per_frame=True,
                       prepend_hist=True).eval()
    torch.manual_seed(0)
    ref = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                         corr_rf="bwd", t_per_frame=True).eval()
    assert m.trainable_param_count() == ref.trainable_param_count()
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randn(2, 8, 8, 8).tanh()
    t = torch.randint(0, T_STEPS, (2, 8))
    with torch.no_grad():
        u1, _ = m.denoise(xt, hist, t)
        hist2 = hist.clone()
        hist2[:, -1] = -hist2[:, -1]      # flip the clean anchor frame
        u2, _ = m.denoise(xt, hist2, t)
    assert u1.shape == (2, 8, 8, 8)
    assert not torch.allclose(u1[:, 0], u2[:, 0]), \
        "frame-0 logits must respond to the clean prepended frame"


def test_corr_head_bilinear_params_and_shapes():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       corr_head="bilinear").eval()
    corr_params = sum(p.numel() for n, p in m.named_parameters()
                      if n.startswith("corr_"))
    assert corr_params == (34 * 64 + 64) + (36 * 64 + 64) + (64 + 1)
    assert not hasattr(m, "corr"), "bilinear must not build the mlp head"
    legacy = DiffusionModel(grid=8, arm="existence")
    assert not hasattr(legacy, "corr_wf"), "legacy must not build bilinear"
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randn(2, 8, 8, 8).tanh()
    with torch.no_grad():
        u, _ = m.denoise(xt, hist, torch.tensor([3, 7]))
    assert u.shape == (2, 8, 8, 8)


def test_spin_scale_and_canvas_tanh_smoke():
    """spin_scale=5 + canvas_tanh + prepend: training loss runs, tf-acc
    valid, rollout returns binary predictions of the right shape."""
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       corr_rf="bwd", corr_head="bilinear", t_per_frame=True,
                       prepend_hist=True, spin_scale=5.0,
                       canvas_tanh=True)
    frames = (torch.randint(0, 2, (4, 1, 16, 8, 8)).float() * 2 - 1)
    loss, acc = m.training_loss(frames)
    assert torch.isfinite(loss) and 0.0 <= acc.item() <= 1.0
    m.eval()
    with torch.no_grad():
        preds = m.rollout(frames[:, :, :8], schedule="staggered")
    assert preds.shape == (4, 8, 8, 8)
    assert set(preds.unique().tolist()) <= {0, 1}


def test_t_frame_slope_offset():
    """t_frame_slope_now shifts per-frame noise levels: with a huge slope
    every frame lands at t=T-1, changing the loss vs slope=0 on identical
    RNG (same base t, same eps draw)."""
    frames = (torch.randint(0, 2, (8, 1, 16, 8, 8)).float() * 2 - 1)
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       t_per_frame=True)
    torch.manual_seed(7)
    m.t_frame_slope_now = 0.0
    loss0, _ = m.training_loss(frames)
    torch.manual_seed(7)
    m.t_frame_slope_now = 100.0
    loss1, _ = m.training_loss(frames)
    assert not torch.allclose(loss0, loss1), "slope must shift the noising"
