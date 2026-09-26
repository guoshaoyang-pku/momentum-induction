"""E12.2 / E12.6 (SETTINGS S14): position-free arms.

`art_twohop --twohop_pos none` and `art_vanilla --pos_enc none` must leave the
registered defaults bit-identical, drop exactly the positional parameters, and
(for the two-hop NoPE arm) make retrieval blind to token order: with no
position tables and unmasked attention, the pooled evidence is a function of
the SET of prefix tokens, so reordering prefix frames cannot change a logit.
"""

import torch

from cawm.models.art_twohop import ArtTwoHop
from cawm.models.art_vanilla import ArtVanilla
from cawm.train import build_model, model_kwargs


def _seeded(cls, **kw):
    torch.manual_seed(11)
    return cls(grid=8, **kw)


def test_twohop_default_is_bit_identical_to_learned():
    a, b = _seeded(ArtTwoHop), _seeded(ArtTwoHop, pos="learned")
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert "tok_pos" in sa and "q_pos" in sa


def test_twohop_nope_drops_only_the_position_tables():
    a, n = _seeded(ArtTwoHop), _seeded(ArtTwoHop, pos="none")
    assert "tok_pos" not in n.state_dict() and "q_pos" not in n.state_dict()
    assert set(a.state_dict()) - set(n.state_dict()) == {"tok_pos", "q_pos"}
    assert a.trainable_param_count() - n.trainable_param_count() == (512 + 64) * 64


def test_twohop_nope_is_blind_to_prefix_order():
    m = _seeded(ArtTwoHop, pos="none").eval()
    x = torch.randint(0, 2, (3, 1, 8, 8, 8)).float() * 2 - 1
    cur = torch.randint(0, 2, (3, 1, 8, 8)).float() * 2 - 1
    with torch.no_grad():
        l0 = m._logits_from(m._tokens(x), cur)
        l1 = m._logits_from(m._tokens(x.flip(2)), cur)      # frames reversed
    assert torch.allclose(l0, l1, atol=1e-5)
    # the learned arm is NOT order-blind (sanity: the test can fail)
    ml = _seeded(ArtTwoHop).eval()
    with torch.no_grad():
        d = (ml._logits_from(ml._tokens(x), cur)
             - ml._logits_from(ml._tokens(x.flip(2)), cur)).abs().max()
    assert d > 1e-4


def test_twohop_nope_runs_forward_and_rollout():
    m = _seeded(ArtTwoHop, pos="none").eval()
    f = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    with torch.no_grad():
        assert m(f).shape == (2, 8, 8, 8)
        r = m.rollout(f[:, :, :8])
    assert r.shape == (2, 8, 8, 8) and r.dtype == torch.uint8


def test_vanilla_nope_has_no_position_structure():
    a, n = _seeded(ArtVanilla, dmodel=64), _seeded(ArtVanilla, dmodel=64, pos_enc="none")
    assert not any(k == "pos" or k.startswith("pos_") for k in n.state_dict())
    assert not any(blk.rope for blk in n.blocks)
    assert a.trainable_param_count() - n.trainable_param_count() == a.ntok * 64
    f = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    with torch.no_grad():
        out = n(f)
        r = n.rollout(f[:, :, :8])
    assert out.shape == (2, 8, 8, 8) and torch.isfinite(out).all()
    assert r.shape == (2, 8, 8, 8)


def test_model_kwargs_defaults_for_old_checkpoints():
    base = dict(grid=8, arm="emergence", head="concat")
    kw = model_kwargs(base)                     # args recorded before E12
    assert kw["twohop_pos"] == "learned" and kw["pos_enc"] == "learned"
    m = build_model(42, model="art_twohop", **kw)
    assert m.pos == "learned"
    m2 = build_model(42, model="art_twohop", **model_kwargs(dict(base, twohop_pos="none")))
    assert m2.pos == "none"
