"""E6.5 ART variants: default OFF is bit-identical; ON variants are sane."""
import torch

from cawm.models.art_induction import ArtInduction
from cawm.train import build_model


def _x(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (B, 1, 16, 8, 8), generator=g).float() * 2 - 1)


def test_defaults_bitcompat_with_registered_art():
    a = build_model(42, model="art", grid=8, arm="emergence")
    b = build_model(42, model="art", grid=8, arm="emergence",
                    art_global=False, art_null_token=False, art_head_hidden=24)
    x = _x()
    assert torch.equal(a(x), b(x))
    assert a.trainable_param_count() == 3798


def test_global_path_shapes_and_params():
    m = build_model(42, model="art", grid=8, arm="emergence", art_global=True)
    x = _x()
    assert m(x).shape == (3, 8, 8, 8)
    assert m.rollout(x[:, :, :8]).shape == (3, 8, 8, 8)
    assert m.trainable_param_count() == 3798 + 36 * 24


def test_global_path_is_evidence_counts():
    """The global vector equals the summed 36-d tokens (per-(ctx,outcome)
    counts on the existence stack)."""
    m = ArtInduction(grid=8, arm="existence", global_path=True)
    x = _x()
    tok = m._tokens(x[:, :, :8])
    g = tok.sum(dim=1)
    assert torch.allclose(g.sum(dim=-1), torch.full((3,), 448.0), atol=1e-3)


def test_null_token_takes_mass_when_no_match():
    m = ArtInduction(grid=8, arm="existence", null_token=True)
    with torch.no_grad():
        m.k_null.fill_(0.5)
        m.v_null.fill_(-3.0)
    x = _x()
    tok = m._tokens(x[:, :, :8])
    q = torch.zeros(3, 1, 18)
    q[:, 0, 17] = 1.0                 # slot (s=1,n=8): rarely present
    present = tok.reshape(3, -1, 36)[..., 2 * 17:2 * 17 + 2].sum(dim=(1, 2)) > 0
    out = m._attend(tok, q)
    # k_null=0.5 -> sink score 10 vs 448 zero-score tokens: e^10/(e^10+448)
    # = 98% of the mass lands on the sink (a learned k_null can push it to 1)
    for b in range(3):
        if not present[b]:
            assert out[b, 0].mean() < -2.8
    assert m.trainable_param_count() == 913 + 36


def test_head_hidden_param_count():
    m = build_model(42, model="art", grid=8, arm="emergence", art_head_hidden=48)
    assert m.trainable_param_count() == 3798 - (36 * 24 + 24 + 24 + 1) + (36 * 48 + 48 + 48 + 1)
