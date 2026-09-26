"""Tests for the LifeGPT config adapter (scripts/cawm/models/lifegpt_config.py)."""

import torch

from scripts.cawm.models.lifegpt_config import LifeGPTConfig


def _random_frames(B=2, T=16, H=8, W=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (B, 1, T, H, W), generator=g)
    return bits.float() * 2.0 - 1.0


def _model(seed=0):
    torch.manual_seed(seed)
    return LifeGPTConfig().float().eval()


def test_param_count():
    model = _model()
    n_train = model.trainable_param_count()
    n_total = model.total_param_count()
    print("lifegpt_cfg trainable params: %d  total params: %d" % (n_train, n_total))
    assert 500_000 <= n_train <= 1_500_000
    assert n_total >= n_train


def test_forward_shape():
    model = _model()
    frames = _random_frames()
    with torch.no_grad():
        logits = model(frames)
    assert logits.shape == (2, 8, 8, 8)
    assert logits.dtype == torch.float32


def test_rollout_shape():
    model = _model()
    frames = _random_frames()
    pred = model.rollout(frames[:, :, :8])
    assert pred.shape == (2, 8, 8, 8)
    assert pred.dtype == torch.uint8
    assert set(pred.unique().tolist()) <= {0, 1}


def test_prefix_consistency():
    """Rollout's first predicted frame from the true 8-frame prefix must match
    forward's j=0 output (CPU, float32, atol=1e-5)."""
    model = _model()
    frames = _random_frames()
    with torch.no_grad():
        fwd_logits_j0 = model(frames)[:, 0]
        roll_logits_j0 = model._logits(frames[:, :, :8])[:, -1]
        roll_bits_j0 = model.rollout(frames[:, :, :8])[:, 0]
    assert torch.allclose(fwd_logits_j0, roll_logits_j0, atol=1e-5)
    assert torch.equal((fwd_logits_j0 > 0).to(torch.uint8), roll_bits_j0)


def test_no_leakage():
    """Perturbing a LATER input frame must not change logits for earlier
    output frames. Output frame 8+j depends only on input frames 0..7+j, so
    flipping input frame 12 may only affect outputs j >= 5."""
    model = _model()
    frames = _random_frames()
    perturbed = frames.clone()
    perturbed[:, :, 12] = -perturbed[:, :, 12]  # flip every cell of frame 12
    with torch.no_grad():
        base = model(frames)
        pert = model(perturbed)
    assert torch.allclose(base[:, :5], pert[:, :5], atol=1e-5), \
        "earlier output frames changed after perturbing a later input frame"
    assert not torch.allclose(base[:, 5:], pert[:, 5:], atol=1e-5), \
        "later output frames should react to the perturbation"


if __name__ == "__main__":
    test_param_count()
    test_forward_shape()
    test_rollout_shape()
    test_prefix_consistency()
    test_no_leakage()
    print("all lifegpt_cfg tests passed")
