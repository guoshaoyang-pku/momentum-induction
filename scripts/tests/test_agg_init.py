"""E4.6: aggregation-init control for the emergence arm (CNN + diffusion).

'ones' must stay bitwise-identical to the registered init; 'random' /
'positive' must change ONLY the aggregation parameters (pyramid + temporal
merge), so a control run is a paired comparison against the registered run
with the same seed.
"""

import torch

from cawm.train import build_model

DIFF = dict(no_cascade=True, corr_rf="bwd", prepend_hist=True, t_per_frame=True,
            spin_scale=5.0, canvas_tanh=True, corr_head="bilinear", frontier=True)
ARMS = [("constructive", dict(head="concat")), ("diffusion", DIFF)]


def _is_agg(k):
    return k.startswith("pyramid") or k.startswith("temporal")


def _sd(model, init, **kw):
    m = build_model(42, model=model, grid=8, arm="emergence", agg_init=init, **kw)
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_ones_is_default_and_bitwise():
    for model, kw in ARMS:
        a = _sd(model, "ones", **kw)
        m = build_model(42, model=model, grid=8, arm="emergence", **kw)  # no flag
        for k, v in m.state_dict().items():
            assert torch.equal(a[k], v), (model, k)
        for conv_k in [k for k in a if k.startswith("pyramid") and k.endswith("weight")]:
            assert torch.all(a[conv_k] == 1.0)
        assert torch.all(a["temporal_w"] == 1.0) and torch.all(a["temporal_b"] == 0.0)


def test_random_and_positive_touch_only_aggregation():
    for model, kw in ARMS:
        ones = _sd(model, "ones", **kw)
        for init in ("random", "positive"):
            sd = _sd(model, init, **kw)
            for k in ones:
                if _is_agg(k):
                    continue
                assert torch.equal(ones[k], sd[k]), (model, init, k)
            assert not torch.equal(ones["pyramid.0.weight"], sd["pyramid.0.weight"])
            assert not torch.equal(ones["temporal_w"], sd["temporal_w"])
            if init == "positive":
                assert torch.all(sd["pyramid.0.weight"] > 0) and torch.all(sd["temporal_w"] > 0)
                assert torch.all(sd["temporal_b"] == 0.0)
            else:
                assert (sd["pyramid.0.weight"] < 0).any() and (sd["temporal_w"] < 0).any()


def test_random_init_forward_runs():
    m = build_model(42, model="constructive", grid=8, arm="emergence", head="concat",
                    agg_init="random")
    x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    y = m(x)
    assert y.shape == (2, 8, 8, 8) and torch.isfinite(y).all()
