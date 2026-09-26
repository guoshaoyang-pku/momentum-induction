"""Unit tests for the Burtsev config adapter (house interface contract)."""
import torch

from cawm.models.burtsev_config import BurtsevConfigWM


def _model():
    # Seed the module init: the model is prefix-consistent by construction, but
    # greedy rollout vs teacher-forced logits can flip on near-tie cells when the
    # untrained weights land near 0; deterministic init keeps the check stable.
    torch.manual_seed(0)
    return BurtsevConfigWM(grid=8, dmodel=64, n_layer=2, n_head=4)


def _traj_pm1(b=2):
    g = torch.Generator().manual_seed(0)
    return (torch.randint(0, 2, (b, 1, 16, 8, 8), generator=g).float() * 2 - 1)


def test_shapes():
    m = _model()
    x = _traj_pm1()
    assert m(x).shape == (2, 8, 8, 8)
    r = m.rollout(x[:, :, :8])
    assert r.shape == (2, 8, 8, 8) and r.dtype == torch.uint8


def test_prefix_consistency():
    m = _model()
    x = _traj_pm1()
    with torch.no_grad():
        fwd_first = m(x)[:, 0]                    # frame 8 from true frames 0..7
        roll_first = m.rollout(x[:, :, :8])[:, 0].float()
    # greedy rollout = sign of the same logits; near-tie cells can flip
    agree = ((fwd_first > 0).float() == roll_first).float().mean()
    assert agree > 0.999  # inclusive mask: exact prefix consistency


def test_leakage():
    m = _model()
    x = _traj_pm1()
    x2 = x.clone()
    x2[:, 0, 10:] = -x2[:, 0, 10:]                # perturb frames 10..15
    with torch.no_grad():
        l1, l2 = m(x), m(x2)
    # output frame 8+j (j=0..1 here) depends only on input frames <= 8+j
    assert torch.allclose(l1[:, :2], l2[:, :2], atol=1e-5)


def test_params():
    m = BurtsevConfigWM(grid=8)
    n = m.trainable_param_count()
    print("burtsev_cfg default params:", n)
    assert 0.2e6 < n < 2.0e6
