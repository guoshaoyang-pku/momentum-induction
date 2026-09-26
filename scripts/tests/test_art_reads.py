"""E21: R-read depth for the pairing-conv ART and the KV-shift model."""
import pytest
import torch

from cawm.train import build_model, model_kwargs

PARAMS = {("art", 1): 3798, ("art", 2): 6192, ("art", 3): 8586, ("art", 4): 10980,
          ("art_kvshift", 1): 2625, ("art_kvshift", 2): 5019,
          ("art_kvshift", 3): 7413, ("art_kvshift", 4): 9807}
KEY = {"art": "art_reads", "art_kvshift": "kv_reads"}


def _build(model, reads, seed=43, grid=8):
    a = dict(seed=seed, model=model, grid=grid, arm="emergence", head="concat",
             **{KEY[model]: reads})
    return build_model(seed, model=model, **model_kwargs(a))


@pytest.mark.parametrize("model,reads", sorted(PARAMS))
def test_param_counts(model, reads):
    assert _build(model, reads).total_param_count() == PARAMS[(model, reads)]


@pytest.mark.parametrize("model", ["art", "art_kvshift"])
def test_extra_reads_leave_first_read_init_unchanged(model):
    one, four = _build(model, 1).state_dict(), _build(model, 4).state_dict()
    for k, v in one.items():
        if not k.startswith("head."):
            assert torch.equal(v, four[k]), k


def test_art_default_reads_is_registered_model():
    a = dict(seed=42, model="art", grid=8, arm="emergence", head="concat")
    m = build_model(42, model="art", **model_kwargs(a))
    n = _build("art", 1, seed=42)
    assert m.reads == 1 and not hasattr(m, "query_update")
    for k, v in m.state_dict().items():
        assert torch.equal(v, n.state_dict()[k])


@pytest.mark.parametrize("model,reads", [("art", 3), ("art_kvshift", 4)])
def test_gradients_causality_and_reload(model, reads):
    m = _build(model, reads, grid=4)
    x = torch.randint(0, 2, (2, 1, 16, 4, 4)).float() * 2 - 1
    m(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)
    m.eval()
    n = _build(model, reads, grid=4)
    n.load_state_dict(m.state_dict())
    n.eval()
    changed = x.clone()
    changed[:, :, 8:] *= -1
    with torch.no_grad():
        assert torch.equal(m(x)[:, 0], m(changed)[:, 0])
        p = m.rollout(x[:, :, :8])
        assert torch.equal(p, n.rollout(x[:, :, :8]))
        assert torch.equal(p[:, 0], (m(x)[:, 0] > 0).to(torch.uint8))
