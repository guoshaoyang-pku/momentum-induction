"""E10 billiard world + per-frame DiffusionVanilla tests (SETTINGS S12)."""

import numpy as np
import torch

from cawm.billiard import (MOVES, _world_params, build_eval_set, overlap_rate,
                           rollout_metrics, sample_frames, simulate_world)
from cawm.models import DiffusionVanilla
from cawm.models.diffusion import T_STEPS


def test_world_determinism():
    a = sample_frames(range(8), 42)
    b = sample_frames(range(8), 42)
    c = sample_frames(range(1, 9), 42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[1:], c)


def test_world_dynamics():
    # single ball: constant velocity on the torus, occupancy rolls exactly
    pos = np.array([[15, 3]])
    vel = np.array([[1, 0]])                       # +y, wraps at the edge
    frames = simulate_world(pos, vel, 15, 16)
    assert frames.shape == (16, 16, 16)
    for k in range(16):
        assert frames[k].sum() == 1
        assert frames[k][(15 + k) % 16, 3] == 1
    # ball count never exceeds n_balls in any frame (OR-occupancy can merge)
    batch = sample_frames(range(64), 42)
    assert (batch.reshape(64, 16, -1).sum(-1) <= 3).all()
    assert (batch.reshape(64, 16, -1).sum(-1) >= 1).all()


def test_world_params_draw_order():
    pos, vel = _world_params(7, 42, 16, 3)
    assert pos.shape == (3, 2) and vel.shape == (3, 2)
    assert len({tuple(p) for p in pos}) == 3       # distinct init cells
    assert all((v != 0).any() for v in vel)        # nonzero king moves
    pos2, vel2 = _world_params(7, 42, 16, 3)
    assert np.array_equal(pos, pos2) and np.array_equal(vel, vel2)


def test_eval_set_sha_stable():
    frames, sha = build_eval_set(16)
    _, sha2 = build_eval_set(16)
    assert sha == sha2 and frames.shape == (16, 16, 16, 16)
    assert 0.0 <= overlap_rate(frames, 3) <= 1.0


def test_metrics():
    truth = np.zeros((4, 8, 4, 4), dtype=np.uint8)
    preds = truth.copy()
    preds[0, 0, 0, 0] = 1                          # one wrong cell in world 0
    m = rollout_metrics(preds, truth)
    assert m["seq_acc"] == 0.75
    assert m["frame_perfect"][0] == 0.75
    assert m["frame_perfect"][1] == 1.0
    assert m["pixel_acc"] > 0.99


def test_vanilla_per_frame_shapes():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    xt = torch.randn(2, 8, 8, 8)
    hist = torch.randn(2, 8, 8, 8)
    tcol = torch.randint(0, T_STEPS, (2, 8))
    u, _ = m.denoise(xt, hist, tcol)
    assert u.shape == (2, 8, 8, 8)
    frames = torch.randn(2, 1, 16, 8, 8)
    loss, acc = m.training_loss(frames, pos_weight=8.0)
    assert loss.dim() == 0 and 0.0 <= float(acc) <= 1.0
    prefix = torch.randn(2, 1, 8, 8, 8)
    p1 = m.rollout(prefix, schedule="uniform")
    p2 = m.rollout(prefix, schedule="frame_ar", commit=True)
    assert p1.shape == p2.shape == (2, 8, 8, 8)
    assert p1.dtype == torch.uint8 and p2.dtype == torch.uint8


def test_vanilla_legacy_path_unchanged():
    # scalar-t denoise / training_loss / uniform rollout stay the legacy path
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8)
    xt, hist = torch.randn(2, 8, 8, 8), torch.randn(2, 8, 8, 8)
    t = torch.randint(0, T_STEPS, (2,))
    u, _ = m.denoise(xt, hist, t)
    assert u.shape == (2, 8, 8, 8)
    loss, acc = m.training_loss(torch.randn(2, 1, 16, 8, 8))
    assert loss.dim() == 0
    p = m.rollout(torch.randn(2, 1, 8, 8, 8), schedule="frame_ar", commit=True)
    assert p.shape == (2, 8, 8, 8)                 # ignored without t_per_frame


def test_frame_ar_commit_order():
    # during frame_ar, every denoise call declares at most one nonzero level,
    # and the active column is nondecreasing (frame k finishes before k+1)
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    seen = []
    orig = m.denoise

    def spy(xt, hist, t):
        seen.append(t.detach().cpu().clone())
        return orig(xt, hist, t)

    m.denoise = spy
    m.rollout(torch.randn(1, 1, 8, 8, 8), schedule="frame_ar", commit=True)
    cols = [int(t[0]. nonzero()[0]) if (t[0] > 0).any() else -1 for t in seen]
    nz = [c for c in cols if c >= 0]
    assert len(seen) == 8 * (T_STEPS - 1)
    assert all(b >= a for a, b in zip(nz, nz[1:]))  # nondecreasing active frame
    assert sorted(set(nz)) == list(range(8))


def test_tiny_overfit_smoke():
    # 120 optimizer steps on a pinned batch must learn: tf pixel accuracy
    # climbs past 0.9 (the pos_weight=8 BCE plateaus ~0.78 even when the
    # model is right, so the accuracy is the informative signal here)
    torch.manual_seed(0)
    np.random.seed(0)
    frames = torch.as_tensor(sample_frames(range(16), 42, grid=8)).float()
    x = (2 * frames - 1)[:, None]
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    acc = 0.0
    for step in range(120):
        loss, acc = m.training_loss(x, pos_weight=8.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < first and float(acc) > 0.9


def test_collision_swaps_velocities():
    # two balls head-on sharing a cell: both annihilate (E10.4 rule)
    pos = np.array([[7, 7], [7, 9]])
    vel = np.array([[0, 1], [0, -1]])              # approach: 7,8 <- 7,8
    frames = simulate_world(pos, vel, 5, 16, collide=True)
    assert frames[0][7, 7] == 1 and frames[0][7, 9] == 1
    assert frames[1][7, 8] == 1 and frames[1].sum() == 1     # merged frame
    assert frames[2].sum() == 0 and frames[5].sum() == 0     # both gone
    # no collision -> both survive, and occupancy differs
    f3 = simulate_world(pos, vel, 5, 16, collide=False)
    assert f3[2].sum() == 2 and not np.array_equal(frames, f3)
    # a third ball elsewhere is unaffected by the annihilation
    pos3 = np.array([[7, 7], [7, 9], [0, 0]])
    vel3 = np.array([[0, 1], [0, -1], [1, 0]])
    f4 = simulate_world(pos3, vel3, 5, 16, collide=True)
    assert f4[2].sum() == 1 and f4[2][2, 0] == 1


def test_collision_stream_reproducible():
    a = sample_frames(range(8), 42, collide=True)
    b = sample_frames(range(8), 42, collide=True)
    assert np.array_equal(a, b)


def test_window_w1_equals_frame_ar():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    prefix = torch.randn(2, 1, 8, 8, 8)
    torch.manual_seed(7)
    a = m.rollout(prefix, schedule="frame_ar", commit=True)
    torch.manual_seed(7)
    b = m.rollout_window(prefix, window=1)
    assert torch.equal(a, b)


def test_window_w8_equals_uniform():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    prefix = torch.randn(2, 1, 8, 8, 8)
    torch.manual_seed(7)
    a = m.rollout(prefix, schedule="uniform")
    torch.manual_seed(7)
    b = m.rollout_window(prefix, window=8)
    assert torch.equal(a, b)


def test_polish_and_inject_shapes():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    prefix = torch.randn(1, 1, 8, 8, 8)
    out = m.rollout_window(prefix, window=1, polish=4)
    assert out.shape == (1, 8, 8, 8) and out.dtype == torch.uint8
    out2 = m.rollout_window(prefix, window=2, polish=4, inject=(3, (2, 5)))
    assert out2.shape == (1, 8, 8, 8)
