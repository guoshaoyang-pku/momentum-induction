"""A6 tests: pairing tokenizer for ArtVanilla (default bit-identical, pair adds
only pair_proj, leakage-safe, prefix-consistent, wired through train.py)."""

import torch

from cawm.models.art_vanilla import ArtVanilla
from cawm.train import build_model, model_kwargs

D64_PARAMS = 165825
PAIR_EXTRA = 10 * 64 + 64  # pair_proj.weight + pair_proj.bias


def _mk(**kw):
    torch.manual_seed(0)
    return ArtVanilla(**kw)


def test_default_is_cell_and_bit_identical():
    # (i) default constructor == tokenizer="cell", param count registered at d64
    a = _mk()
    b = _mk(tokenizer="cell")
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa.keys()) == list(sb.keys())
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    assert a.tokenizer == "cell"
    assert a.trainable_param_count() == D64_PARAMS
    assert b.trainable_param_count() == D64_PARAMS


def test_pair_adds_only_pair_proj():
    # (ii) exactly 10*64+64 extra params; keys = cell keys + pair_proj.*
    a = _mk(tokenizer="cell")
    b = _mk(tokenizer="pair")
    assert b.trainable_param_count() - a.trainable_param_count() == PAIR_EXTRA
    extra = set(b.state_dict().keys()) - set(a.state_dict().keys())
    assert extra == {"pair_proj.weight", "pair_proj.bias"}
    assert set(a.state_dict().keys()) - set(b.state_dict().keys()) == set()


def test_leakage_future_flip_leaves_past_unchanged():
    # (iii) flipping token j > i leaves _encode(toks)[:, :i+1] unchanged
    torch.manual_seed(1)
    m = _mk(tokenizer="pair").eval()
    toks = torch.randint(0, 2, (2, m.ntok))
    with torch.no_grad():
        base = m._encode(toks)
    # incl. i and j inside the same frame: (100,101) frame 1, (513,540) frame 8
    pairs = [(0, 1), (10, 63), (100, 101), (100, 500), (513, 540), (700, 1023)]
    for i, j in pairs:
        assert j > i
        t2 = toks.clone()
        t2[:, j] = 1 - t2[:, j]
        with torch.no_grad():
            h2 = m._encode(t2)
        assert torch.allclose(h2[:, :i + 1], base[:, :i + 1],
                              atol=1e-6, rtol=0), (i, j)


def test_prefix_consistency():
    # (iv) _encode(toks[:, :L]) == _encode(toks)[:, :L]
    torch.manual_seed(3)
    toks = torch.randint(0, 2, (2, 16 * 8 * 8))
    for tok_mode in ("cell", "pair"):
        m = _mk(tokenizer=tok_mode).eval()
        with torch.no_grad():
            full = m._encode(toks)
            for L in (64, 100, 513, 1023):
                pre = m._encode(toks[:, :L])
                # attention kernels vary at the 1e-6 level with sequence
                # length in BOTH modes (the registered "cell" model included),
                # so the bound is 1e-5: leakage would show up at O(1).
                assert torch.allclose(pre, full[:, :L],
                                      atol=1e-5, rtol=0), (tok_mode, L)


def test_train_wiring():
    # (v) model_kwargs coerces missing tokenizer to "cell"; build_model builds pair
    base = {"grid": 8, "arm": "emergence", "head": "concat", "dmodel": 64}
    kw = model_kwargs(base)  # no 'tokenizer' key
    assert kw["tokenizer"] == "cell"
    kw2 = model_kwargs({**base, "tokenizer": "pair"})
    assert kw2["tokenizer"] == "pair"
    m = build_model(seed=1, model="art_vanilla", **kw2)
    assert hasattr(m, "pair_proj")
    assert m.tokenizer == "pair"


def test_forward_and_rollout_shapes():
    # (vi) forward (B,8,8,8) float logits; rollout (B,8,8,8) uint8
    torch.manual_seed(2)
    m = _mk(tokenizer="pair").eval()
    frames = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    out = m(frames)
    assert out.shape == (2, 8, 8, 8)
    assert out.dtype == torch.float32
    prefix = torch.randint(0, 2, (2, 1, 8, 8, 8)).float() * 2 - 1
    r = m.rollout(prefix)
    assert r.shape == (2, 8, 8, 8)
    assert r.dtype == torch.uint8
