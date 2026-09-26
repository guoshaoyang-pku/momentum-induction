"""E3.8 chain-training tests.

  default off      — DiffusionModel() has chain_nfe None; legacy rollout path
                     (full 49-level chain from N(0,1)) is untouched.
  chain_levels(10) — exactly 10 unique descending levels in 1..49.
  chain_loss       — runs, finite, gradient reaches trainable params; zeros
                     init is deterministic, noise init is stochastic.
  rollout w/ chain — a chain-configured model evals through its own chain
                     (zeros: deterministic; noise: stochastic); shapes intact.
"""

import pytest
import torch

from cawm.models import DiffusionModel
from cawm.models.diffusion import T_STEPS, chain_levels


# ------------------------------------------------------------------ default

def test_default_off_is_legacy():
    m = DiffusionModel(grid=8, arm="existence")
    assert m.chain_nfe is None
    assert m.chain_init == "noise"
    # default chain_levels behaviour unchanged
    assert chain_levels(T_STEPS - 1) == list(range(T_STEPS - 1, 0, -1))


def test_chain_levels_ten():
    lv = chain_levels(10)
    assert len(lv) == len(set(lv)) == 10
    assert lv == sorted(lv, reverse=True)
    assert all(1 <= t <= T_STEPS - 1 for t in lv)


# --------------------------------------------------------------- chain_loss

def _batch(b=4, grid=8):
    from cawm.data import StreamDataset
    stream = StreamDataset(42, half="train", length=b * 8, grid=grid)
    frames = stream.get_batch(list(range(b)))["frames"].float()
    return (2 * frames - 1).unsqueeze(1)


def test_chain_loss_runs_and_backprops():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="emergence", no_cascade=True,
                       corr_rf="bwd", prepend_hist=True, t_per_frame=True,
                       spin_scale=5.0, canvas_tanh=True, corr_head="bilinear",
                       chain_nfe=10, chain_init="zeros")
    x = _batch()
    loss, acc = m.chain_loss(x, aux_weight=0.2)
    assert torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0
    loss.backward()
    grads = [p.grad for p in m.parameters()
             if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_chain_zeros_deterministic_noise_stochastic():
    torch.manual_seed(0)
    mz = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                        chain_nfe=10, chain_init="zeros").eval()
    mn = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                        chain_nfe=10, chain_init="noise").eval()
    x = _batch()
    with torch.no_grad():
        lz1, _ = mz.chain_loss(x)
        lz2, _ = mz.chain_loss(x)
        torch.manual_seed(1)
        ln1, _ = mn.chain_loss(x)
        torch.manual_seed(2)
        ln2, _ = mn.chain_loss(x)
    assert float(lz1) == float(lz2)          # zeros start: fully deterministic
    assert float(ln1) != float(ln2)          # noise start: stochastic


def test_chain_aux_weight_zero_final_only():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       chain_nfe=10, chain_init="zeros").eval()
    x = _batch()
    with torch.no_grad():
        l_final, _ = m.chain_loss(x, aux_weight=0.0)
        l_aux, _ = m.chain_loss(x, aux_weight=0.2)
    assert float(l_aux) > float(l_final)     # aux adds the intermediate terms


# ------------------------------------------------------------------- rollout

def test_rollout_uses_chain_config():
    torch.manual_seed(0)
    mz = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                        prepend_hist=True, spin_scale=5.0, canvas_tanh=True,
                        corr_head="bilinear", corr_rf="bwd",
                        chain_nfe=10, chain_init="zeros").eval()
    prefix = torch.randn(2, 1, 8, 8, 8)
    p1 = mz.rollout(prefix)
    p2 = mz.rollout(prefix)
    assert p1.shape == (2, 8, 8, 8) and p1.dtype == torch.uint8
    assert torch.equal(p1, p2)               # zeros chain: deterministic eval


# ---------------------------------------------------------------- frontier

def test_frontier_default_off():
    m = DiffusionModel(grid=8, arm="existence")
    assert m.frontier is False


def test_frontier_loss_runs_and_backprops():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="emergence", no_cascade=True,
                       corr_rf="bwd", prepend_hist=True, t_per_frame=True,
                       spin_scale=5.0, canvas_tanh=True, corr_head="bilinear",
                       frontier=True)
    x = _batch()
    loss, acc = m.frontier_loss(x)
    assert torch.isfinite(loss) and 0.0 <= float(acc) <= 1.0
    loss.backward()
    grads = [p.grad for p in m.parameters()
             if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_frontier_rollout_autoroutes():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       corr_rf="bwd", prepend_hist=True, t_per_frame=True,
                       spin_scale=5.0, canvas_tanh=True, corr_head="bilinear",
                       frontier=True).eval()
    prefix = torch.randn(2, 1, 8, 8, 8)
    torch.manual_seed(3)
    p_auto = m.rollout(prefix)                        # auto frame_ar+commit
    torch.manual_seed(3)
    p_expl = m.rollout(prefix, schedule="frame_ar", commit=True)
    assert p_auto.shape == (2, 8, 8, 8)
    assert torch.equal(p_auto, p_expl)


def test_frame_ar_commit_off_is_legacy():
    torch.manual_seed(0)
    m = DiffusionModel(grid=8, arm="existence", no_cascade=True,
                       corr_rf="bwd", prepend_hist=True, t_per_frame=True,
                       spin_scale=5.0, canvas_tanh=True,
                       corr_head="bilinear").eval()
    prefix = torch.randn(2, 1, 8, 8, 8)
    torch.manual_seed(5)
    p1 = m.rollout(prefix, schedule="frame_ar")
    torch.manual_seed(5)
    p2 = m.rollout(prefix, schedule="frame_ar", commit=False)
    assert torch.equal(p1, p2)
