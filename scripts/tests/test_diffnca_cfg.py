import torch

from cawm.models.diffnca_config import DiffNCACfg


def test_shapes_and_params():
    torch.manual_seed(0)
    m = DiffNCACfg(grid=8)
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randint(0, 2, (2, 8, 8, 8)).float() * 2 - 1
    t = torch.full((2,), 49, dtype=torch.long)
    u, v = m.denoise(xt, hist, t)
    assert u.shape == (2, 8, 8, 8) and v.shape == (2, 8, 8, 8)
    n = m.trainable_param_count()
    assert 500_000 <= n <= 1_500_000, n


def test_per_frame_t_and_single_frame():
    torch.manual_seed(0)
    m = DiffNCACfg(grid=8)
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randint(0, 2, (2, 8, 8, 8)).float() * 2 - 1
    t2 = torch.randint(1, 50, (2, 8))
    u, _ = m.denoise(xt, hist, t2)
    assert u.shape == (2, 8, 8, 8)
    # single-frame path (eval._tf_forward): xt (B,H,W), hist (B,8,H,W)
    u1, _ = m.denoise(xt[:, 0], hist, torch.full((2,), 49, dtype=torch.long))
    assert u1.shape == (2, 8, 8)


def test_training_loss_and_rollout():
    torch.manual_seed(0)
    m = DiffNCACfg(grid=8)
    frames = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    loss, acc = m.training_loss(frames)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0
    m.eval()
    a = m.rollout(frames[:, :, :8])
    b = m.rollout(frames[:, :, :8])
    assert a.dtype == torch.uint8
    assert a.shape == (2, 8, 8, 8)
    assert set(a.unique().tolist()) <= {0, 1}
