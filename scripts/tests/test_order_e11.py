"""E11 (SETTINGS S13): denoiser-depth flag and the pending-frame level of the
frame-ordered sampler for the bidirectional DiffusionVanilla."""

import torch

from cawm.models import DiffusionVanilla
from cawm.models.diffusion import T_STEPS, chain_levels
from cawm.train import build_model, model_kwargs

TOP = chain_levels(T_STEPS - 1)[0]


def _kw(**over):
    base = dict(grid=8, arm="emergence", head="concat")
    base.update(over)
    return model_kwargs(base)


def test_default_depth_state_dict_unchanged():
    # registered generic denoiser: no extra layers, same keys and init
    a = build_model(42, model="diffusion_vanilla", **_kw())
    b = DiffusionVanilla(grid=8, arm="emergence", head="concat")
    assert set(a.state_dict()) == set(b.state_dict())
    assert not any(k.startswith("extra.") for k in a.state_dict())
    assert a.trainable_param_count() == 85697
    torch.manual_seed(42)
    c = DiffusionVanilla(grid=8, arm="emergence", head="concat")
    for k, v in a.state_dict().items():
        assert torch.equal(v, c.state_dict()[k])


def test_depth8_adds_layers():
    m = build_model(42, model="diffusion_vanilla",
                    **_kw(vanilla_depth=8, t_per_frame=True))
    assert m.n_layers == 8 and m.t_per_frame
    assert sum(k.startswith("extra.") for k in m.state_dict()) == 8  # 4 w + 4 b


def test_old_ckpt_args_resolve_to_depth4():
    # a checkpoint written before the flag existed has no vanilla_depth key
    kw = model_kwargs(dict(grid=8, arm="emergence", head="concat"))
    assert kw["vanilla_depth"] == 4


def _spy_levels(m, **rollout_kw):
    seen = []
    orig = m.denoise

    def spy(xt, hist, t):
        seen.append(t.detach().cpu().clone())
        return orig(xt, hist, t)

    m.denoise = spy
    return seen


def test_pending_levels_frame_ar():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    prefix = torch.randn(1, 1, 8, 8, 8)
    seen = _spy_levels(m)
    m.rollout(prefix, schedule="frame_ar", commit=True, pending="noise")
    per_frame = T_STEPS - 1
    for call, t in enumerate(seen):
        k = call // per_frame
        row = t[0]
        assert (row[:k] == 0).all()                    # committed: clean
        assert (row[k + 1:] == TOP).all()              # pending: top level
    seen.clear()
    m.rollout(prefix, schedule="frame_ar", commit=True)   # legacy default
    for call, t in enumerate(seen):
        k = call // per_frame
        assert (t[0][k + 1:] == 0).all()


def test_pending_noise_window_endpoints():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    prefix = torch.randn(2, 1, 8, 8, 8)
    torch.manual_seed(7)
    a = m.rollout(prefix, schedule="frame_ar", commit=True, pending="noise")
    torch.manual_seed(7)
    b = m.rollout_window(prefix, window=1, pending="noise")
    assert torch.equal(a, b)
    torch.manual_seed(7)
    c = m.rollout(prefix, schedule="uniform")
    torch.manual_seed(7)
    d = m.rollout_window(prefix, window=8, pending="noise")
    assert torch.equal(c, d)


def test_pending_window_levels():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    seen = _spy_levels(m)
    m.rollout_window(torch.randn(1, 1, 8, 8, 8), window=2, pending="noise")
    S = T_STEPS - 1
    # first call: frames 0,1 active at the top level, frames 2..7 pending
    assert (seen[0][0] == TOP).all()
    # call S: frames 0,1 committed (0), frames 2,3 start at the top level,
    # frames 4..7 still pending at the top level
    row = seen[S][0]
    assert (row[:2] == 0).all() and (row[2:] == TOP).all()


def test_budget_matched_call_counts():
    # E11.4: total denoiser calls per rollout never exceed the budget
    from eval_order_profile import budget_nfe
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8, t_per_frame=True)
    n = [0]
    orig = m.denoise

    def counted(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    m.denoise = counted
    prefix = torch.randn(1, 1, 8, 8, 8)
    for B in (8, 16, 24, 40, 49):
        for spec, fn in ((dict(schedule="uniform"), m.rollout),
                         (dict(schedule="frame_ar", commit=True, pending="noise"),
                          m.rollout),
                         (dict(window=2, pending="noise"), m.rollout_window),
                         (dict(window=4, pending="noise"), m.rollout_window)):
            n[0] = 0
            fn(prefix, nfe=budget_nfe(spec, B), **spec)
            assert n[0] <= B
    assert budget_nfe(dict(schedule="frame_ar"), 49) == 6
    assert budget_nfe(dict(schedule="uniform"), 49) == 49
    assert budget_nfe(dict(window=2), 49) == 12


def test_frame_stride_stream():
    # E11.5: stride 1 is the registered stream bitwise; stride s shows every
    # s-th generation of the same initial board
    import numpy as np
    from cawm import rules as R
    from cawm.data import sample_batch
    from cawm.simulate import simulate
    idx = range(6)
    r1, f1, c1 = sample_batch(idx, 42, None, rule_override=R.GOL_RULE)
    r0, f0, c0 = sample_batch(idx, 42, None, rule_override=R.GOL_RULE, stride=1)
    assert np.array_equal(f1, f0) and np.array_equal(c1, c0)
    r2, f2, _ = sample_batch(idx, 42, None, rule_override=R.GOL_RULE, stride=2)
    long = simulate(r1, f1[:, 0], 30)
    assert np.array_equal(f2, long[:, ::2]) and f2.shape == f1.shape
    assert np.array_equal(f2[:, 0], f1[:, 0])
