"""Tests for the AutomataGPT-style config adapter (no-RM variant).

Same three checks as the LifeGPT adapter: shapes, prefix consistency
(atol=1e-5), and no target-frame leakage through the causal mask.
"""

import torch

from scripts.cawm.models.automatagpt_config import AutomataGPTConfig

T_TOTAL = 16
T_PREFIX = 8


def _model():
    torch.manual_seed(0)
    model = AutomataGPTConfig()
    model.eval()
    return model


def _frames(batch=2, grid=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (batch, 1, T_TOTAL, grid, grid), generator=g)
    return bits.float().mul(2.0).sub(1.0)


def test_shapes_and_param_budget():
    model = _model()
    frames = _frames()
    with torch.no_grad():
        logits = model(frames)
    assert logits.shape == (2, T_PREFIX, 8, 8)

    prefix = frames[:, :, :T_PREFIX]
    roll = model.rollout(prefix)
    assert roll.shape == (2, T_PREFIX, 8, 8)
    assert roll.dtype == torch.uint8
    assert set(torch.unique(roll).tolist()) <= {0, 1}

    n = model.trainable_param_count()
    assert 500_000 <= n <= 1_500_000, n
    assert model.total_param_count() >= n


def test_prefix_consistency():
    # Output index j conditions only on frames 0..7+j: perturbing later frames
    # must leave logits[:, :j+1] unchanged, and the first rollout step must
    # agree with the first teacher-forced prediction (both see only the
    # 8-frame prefix).
    model = _model()
    frames = _frames(seed=2)
    with torch.no_grad():
        base = model(frames)
        for k in range(1, T_PREFIX):
            alt = frames.clone()
            alt[:, :, T_PREFIX + k :] *= -1.0
            out = model(alt)
            assert torch.allclose(base[:, :k], out[:, :k], atol=1e-5), k

        roll = model.rollout(frames[:, :, :T_PREFIX])
    assert torch.equal((base[:, 0] > 0).to(torch.uint8), roll[:, 0])


def test_no_target_leakage():
    # Prediction for frame 8+j must not see frame 8+j itself: flipping that
    # target frame leaves logits[:, j] unchanged.
    model = _model()
    frames = _frames(seed=3)
    with torch.no_grad():
        base = model(frames)
        for j in range(T_PREFIX):
            alt = frames.clone()
            alt[:, :, T_PREFIX + j] *= -1.0
            out = model(alt)
            assert torch.allclose(base[:, j], out[:, j], atol=1e-5), j
