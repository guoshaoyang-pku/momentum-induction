"""Tests for ArtTemporal (tiny CPU configs, grid=4, batch 1-2).

Parent-integration tests (registry / training loop) are handled at the
repo root; here we test the bounded extension itself, including a
separate state_dict reconstruction test.
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Direct module import (NOT through the package __init__).
from cawm.models.art_temporal import ArtTemporal, _TEMPORAL_MASKS, TEMPORAL_MODES

GRID = 4
VARIANTS = list(itertools.product(TEMPORAL_MODES, ("learned", "none")))


def make(temporal, pos, seed=0):
    torch.manual_seed(seed)
    return ArtTemporal(grid=GRID, temporal=temporal, pos=pos)


def rand_seq(B=1, seed=123):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (B, 1, 16, GRID, GRID), generator=g).float()


def frames_ramp(B=1):
    f = torch.zeros(B, 1, 8, GRID, GRID)
    for t in range(8):
        f[:, :, t] = float(t + 1)
    return f


# ----------------------------------------------------------------------
# Identical allocation / RNG across all 8 variants
# ----------------------------------------------------------------------
def test_identical_state_and_rng_across_variants():
    ref_sd = None
    ref_draw = None
    ref_keys = None
    for temporal, pos in VARIANTS:
        m = make(temporal, pos, seed=0)
        draw = torch.rand(8)  # probes RNG state after construction
        sd = m.state_dict()
        if ref_sd is None:
            ref_sd, ref_draw, ref_keys = sd, draw, set(sd.keys())
        else:
            assert set(sd.keys()) == ref_keys
            for k in ref_keys:
                assert sd[k].shape == ref_sd[k].shape, k
                assert torch.equal(sd[k], ref_sd[k]), k
            assert torch.equal(draw, ref_draw)


def test_equal_param_counts_and_name():
    counts = set()
    for temporal, pos in VARIANTS:
        m = make(temporal, pos)
        counts.add(sum(p.numel() for p in m.parameters()))
        assert m.name == "art_temporal"
        assert m.tok_proj.in_features == 36
    assert len(counts) == 1


# ----------------------------------------------------------------------
# Exact window construction, incl. boundaries (no temporal wrap)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("temporal", list(TEMPORAL_MODES))
def test_memory_windows_exact(temporal):
    m = make(temporal, "learned")
    B = 2
    f = frames_ramp(B)
    w = m._memory_windows(f)
    assert w.shape == (B * 8, 1, 3, GRID, GRID)
    w = w.reshape(B, 8, 1, 3, GRID, GRID)
    mp, mc, mn = _TEMPORAL_MASKS[temporal]
    for t in range(8):
        prev = float(t) if t > 0 else 0.0        # zero-filled at t=0, no wrap
        cur = float(t + 1)
        nxt = float(t + 2) if t < 7 else 0.0     # zero-filled at t=7, no wrap
        expect = (prev * mp, cur * mc, nxt * mn)
        for b in range(B):
            for s in range(3):
                assert torch.all(w[b, t, 0, s] == expect[s]), (t, s)


def test_no_temporal_wrap_symmetric_boundaries():
    m = make("symmetric", "learned")
    f = frames_ramp(1)
    w = m._memory_windows(f).reshape(1, 8, 1, 3, GRID, GRID)
    assert torch.all(w[0, 0, 0, 0] == 0.0)  # prev of frame 0 is zero, not frame 7
    assert torch.all(w[0, 7, 0, 2] == 0.0)  # next of frame 7 is zero, not frame 0


# ----------------------------------------------------------------------
# Memory shapes
# ----------------------------------------------------------------------
def test_memory_shapes():
    m = make("symmetric", "learned")
    B = 2
    prefix = rand_seq(B)[:, :, :8]
    wins = m._memory_windows(prefix)
    assert wins.shape == (B * 8, 1, 3, GRID, GRID)
    codes = m._memory_codes(prefix)
    assert codes.shape == (B, 8 * GRID * GRID, 36)


# ----------------------------------------------------------------------
# First prediction invariant to flipping frames 8..15
# ----------------------------------------------------------------------
def test_first_prediction_invariant_to_future_frames():
    m = make("forward", "learned")
    m.eval()
    full = rand_seq(2)
    alt = full.clone()
    alt[:, :, 8:] = 1.0 - alt[:, :, 8:]
    with torch.no_grad():
        out1 = m(full)
        out2 = m(alt)
    assert out1.shape == (2, 8, GRID, GRID)
    assert torch.equal(out1[:, 0], out2[:, 0])


# ----------------------------------------------------------------------
# forward first step vs explicit first rollout logit (frozen tokens)
# ----------------------------------------------------------------------
def test_forward_matches_explicit_first_logit():
    m = make("center", "learned")
    m.eval()
    full = rand_seq(1)
    with torch.no_grad():
        out = m(full)
        tok = m._tokens(full[:, :, :8])
        l0 = m._logits_from(tok, full[:, :, 7])
    assert torch.allclose(l0.reshape(out[:, 0].shape), out[:, 0], atol=1e-6)


# ----------------------------------------------------------------------
# Gradients: masked temporal conv slices are exactly zero;
# enabled slices receive some gradient.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("temporal,zero_slices,nonzero_slices", [
    ("center", (0, 2), (1,)),
    ("forward", (0,), (1, 2)),
    ("backward", (2,), (0, 1)),
    ("symmetric", (), (0, 1, 2)),
])
def test_masked_temporal_slices_have_zero_grad(temporal, zero_slices, nonzero_slices):
    m = make(temporal, "learned")
    full = rand_seq(1, seed=7)
    out = m(full)
    out.sum().backward()
    g = m.mem_front[0].weight.grad  # (3, 1, 3, 3, 3): out, in, kT, kH, kW
    assert g is not None
    for s in zero_slices:
        assert torch.all(g[:, :, s] == 0), s
    for s in nonzero_slices:
        assert g[:, :, s].abs().sum().item() > 0, s


# ----------------------------------------------------------------------
# PE off: positional parameters allocated but receive no gradient
# ----------------------------------------------------------------------
def test_positional_off_no_grad():
    m = make("center", "none")
    # Parameters are still allocated (equal allocation across variants) ...
    assert isinstance(m.tok_pos, torch.nn.Parameter)
    assert isinstance(m.q_pos, torch.nn.Parameter)
    full = rand_seq(1, seed=3)
    out = m(full)
    out.sum().backward()
    # ... but unused, so no gradient flows into them.
    for p in (m.tok_pos, m.q_pos):
        assert p.grad is None or torch.all(p.grad == 0)
    # Some enabled parameters do get gradient.
    assert m.tok_proj.weight.grad is not None
    assert m.tok_proj.weight.grad.abs().sum().item() > 0


def test_positional_on_has_grad():
    m = make("center", "learned")
    full = rand_seq(1, seed=3)
    out = m(full)
    out.sum().backward()
    assert m.tok_pos.grad is not None
    assert m.tok_pos.grad.abs().sum().item() > 0


# ----------------------------------------------------------------------
# Separate reconstruction test: state_dict round-trip reproduces outputs
# ----------------------------------------------------------------------
def test_state_dict_reconstruction():
    m1 = make("forward", "learned", seed=0)
    m2 = make("forward", "learned", seed=99)
    m1.eval()
    m2.eval()
    full = rand_seq(1)
    with torch.no_grad():
        base = m1(full)
        pre = m2(full)
    assert not torch.equal(base, pre)
    missing = m2.load_state_dict(m1.state_dict())
    assert not missing.missing_keys and not missing.unexpected_keys
    with torch.no_grad():
        post = m2(full)
    assert torch.equal(post, base)


def test_training_checkpoint_reconstruction():
    from cawm.train import build_model, model_kwargs
    for support in TEMPORAL_MODES:
        args = dict(grid=4, arm="emergence", head="concat", temporal_support=support,
                    twohop_pos="none", twohop_d=64, twohop_temp="learned")
        m = build_model(42, model="art_temporal", **model_kwargs(args))
        assert m.temporal == support and m.pos == "none"
        rebuilt = build_model(42, model="art_temporal", **model_kwargs(args))
        rebuilt.load_state_dict(m.state_dict())
