"""Baseline-model tests (§1.4): causality for art_vanilla, parameter counts,
overfit smoke for the three non-diffusion baselines, and the diffusion_vanilla
marginal Monte-Carlo.
"""

import math

import numpy as np
import pytest
import torch

from cawm import rules as R
from cawm.data import StreamDataset
from cawm.models import ARCNN, ArtTwoHop, ArtVanilla, DiffusionVanilla
from cawm.models.diffusion import RETAIN, SIGMA1
from cawm.train import build_model, model_kwargs


# ---------------------------------------------------------------- art_vanilla

def test_art_vanilla_causality():
    """Perturbing a future token must not change past logits (causal mask)."""
    torch.manual_seed(0)
    m = ArtVanilla(grid=8, dmodel=64).eval()
    x = torch.randint(0, 2, (1, 1, 16, 8, 8)).float() * 2 - 1
    with torch.no_grad():
        toks = ((x[:, 0] + 1) / 2).long().reshape(1, -1)
        h1 = m._encode(toks)
        toks2 = toks.clone()
        toks2[0, 900] = 1 - toks2[0, 900]      # flip a future token
        h2 = m._encode(toks2)
    # hidden states at positions < 900 must be identical
    assert torch.allclose(h1[0, :900], h2[0, :900]), "causality violated"
    assert not torch.allclose(h1[0, 900], h2[0, 900])


def test_art_vanilla_shapes_and_sizes():
    for d, layers in ((64, 2), (128, 4), (192, 6)):
        m = ArtVanilla(grid=8, dmodel=d)
        assert len(m.blocks) == layers
        x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
        assert m(x).shape == (2, 8, 8, 8)
        preds = m.rollout(x[:, :, :8])
        assert preds.shape == (2, 8, 8, 8)
        assert set(preds.unique().tolist()) <= {0, 1}


def test_art_vanilla_depth_head_override_bit_identical_at_default():
    """E8.4 added --nlayers/--nhead. Their defaults must reproduce the
    registered width->depth mapping (d64/L2, d128/L4, d192/L6, d256/L8) and the
    4-head count bit-for-bit, or every existing ladder row silently changes
    meaning. Same compat contract as the recipe-v2 flags (SETTINGS.md §9).

    Both sides are built under the same manual_seed: two constructions in a row
    otherwise draw different values from the global RNG for `pos`, which would
    make this test compare seeds rather than architectures.
    """
    def mk(**kw):
        torch.manual_seed(1234)
        return ArtVanilla(grid=8, **kw)

    for d, layers in ((64, 2), (128, 4), (192, 6), (256, 8)):
        a = mk(dmodel=d)
        b = mk(dmodel=d, nlayers=None, nhead=None)
        assert (a.nlayers, a.nhead) == (layers, 4) == (b.nlayers, b.nhead)
        sa, sb = a.state_dict(), b.state_dict()
        assert set(sa) == set(sb)
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # the override itself must actually change the architecture
    for d, L, H in ((192, 2, 4), (192, 2, 2), (64, 6, 4)):
        m = mk(dmodel=d, nlayers=L, nhead=H)
        assert (m.nlayers, m.nhead) == (L, H)
        assert len(m.blocks) == L
        assert m.blocks[0].attn.num_heads == H


def test_art_vanilla_pos_enc_default_bit_identical_and_arms_differ():
    """E7.2 added --pos_enc. 'learned' must leave the module tree exactly as
    registered (single `pos` parameter, non-rotary blocks) so no existing
    ladder row changes meaning; the two new modes must actually differ, and all
    three must produce finite logits and rollouts of the registered shape.
    """
    def mk(**kw):
        torch.manual_seed(7)
        return ArtVanilla(grid=8, dmodel=64, **kw)

    a = mk()
    b = mk(pos_enc="learned")
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert "pos" in sa and not any(k.startswith("pos_") for k in sa)
    assert not any(blk.rope for blk in a.blocks)

    fac, rope = mk(pos_enc="factorized"), mk(pos_enc="rope")
    # factorized trades the (1024,d) table for three small ones
    assert "pos" not in fac.state_dict()
    assert fac.pos_frame.num_embeddings == 16
    assert fac.pos_row.num_embeddings == fac.pos_col.num_embeddings == 8
    # rope adds no positional parameters at all and rotates every block, so its
    # count is exactly the learned arm's minus the (ntok, d) absolute table
    assert all(blk.rope for blk in rope.blocks)
    assert not any(k == "pos" or k.startswith("pos_") for k in rope.state_dict())
    assert (a.trainable_param_count() - rope.trainable_param_count()
            == 16 * 8 * 8 * 64)
    # factorized trades that table for 16 + 8 + 8 rows of width d
    assert fac.trainable_param_count() == (
        a.trainable_param_count() - 16 * 8 * 8 * 64 + (16 + 8 + 8) * 64)

    x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    for m in (a, fac, rope):
        m.eval()
        out = m(x)
        assert out.shape == (2, 8, 8, 8) and torch.isfinite(out).all()
        with torch.no_grad():
            r = m.rollout(x[:, :, :8])
        assert r.shape == (2, 8, 8, 8) and r.dtype == torch.uint8
        assert set(r.unique().tolist()) <= {0, 1}


def test_model_kwargs_rebuilds_the_trained_architecture():
    """The eval path must rebuild exactly what training built.

    Regression for the 2026-09-20 defect: watch_eval.py hand-copied its
    build_model kwargs, E8.4/E7.2 added `nlayers`/`nhead`/`pos_enc` to train.py
    only, and the watcher therefore rebuilt the *default* ladder. Every E8.4 and
    E7.2 checkpoint trained fine and then failed to load ("Missing key(s):
    blocks.2...blocks.5"), i.e. was permanently unscorable. Both paths now go
    through model_kwargs(), and this test pins the contract.
    """
    # args as train.py records them for the three override configurations
    base = dict(grid=8, arm="emergence", head="concat", dmodel=192,
                seed=42, model="art_vanilla")
    cases = [
        dict(nlayers=2, nhead=4, pos_enc="learned"),   # E8.4 (a)
        dict(nlayers=2, nhead=2, pos_enc="learned"),   # E8.4 (b)
        dict(nlayers=6, nhead=4, pos_enc="learned"),   # E8.4 (c)
        dict(nlayers=None, nhead=None, pos_enc="rope"),        # E7.2
        dict(nlayers=None, nhead=None, pos_enc="factorized"),  # E7.2
    ]
    for over in cases:
        cargs = dict(base, **over)
        trained = build_model(42, model="art_vanilla", **model_kwargs(cargs))
        # the watcher's path: same mapping, fresh build, must load the state
        rebuilt = build_model(42, model="art_vanilla", **model_kwargs(cargs))
        rebuilt.load_state_dict(trained.state_dict())   # would raise on mismatch
        assert (rebuilt.trainable_param_count()
                == trained.trainable_param_count())
        assert len(rebuilt.blocks) == (over["nlayers"] or 6)
        assert rebuilt.blocks[0].attn.num_heads == (over["nhead"] or 4)

    # a checkpoint recorded before the pos_enc flag existed has no such key;
    # None must coerce to the registered default rather than reaching the
    # ArtVanilla assert
    legacy = dict(base, nlayers=None, nhead=None, pos_enc=None)
    m = build_model(42, model="art_vanilla", **model_kwargs(legacy))
    assert m.pos_enc == "learned"
    assert "pos" in m.state_dict()

    # the old hand-copied list is what diverged; pin that the shared mapping
    # actually carries the three override keys
    kw = model_kwargs(dict(base, nlayers=2, nhead=2, pos_enc="rope"))
    assert (kw["nlayers"], kw["nhead"], kw["pos_enc"]) == (2, 2, "rope")
    assert {"twohop_temp", "twohop_d", "chain_init"} <= set(kw)


# ------------------------------------------------------------------- arcnn

def test_arcnn_param_count():
    m = ARCNN(grid=8)
    # 9*64*9+64 + 3*(64*64*9+64) + 64*1*1+1 = 5248 + 3*36928 + 65
    expect = (9 * 64 * 9 + 64) + 3 * (64 * 64 * 9 + 64) + (64 + 1)
    assert m.trainable_param_count() == expect
    assert 60_000 < m.trainable_param_count() < 130_000   # ~75k ballpark


def test_arcnn_shapes():
    m = ARCNN(grid=8)
    x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    assert m(x).shape == (2, 8, 8, 8)
    preds = m.rollout(x[:, :, :8])
    assert preds.shape == (2, 8, 8, 8)
    assert set(preds.unique().tolist()) <= {0, 1}


# --------------------------------------------------------------- art_twohop

def test_art_twohop_shapes_and_pe():
    m = ArtTwoHop(grid=8)
    assert m.tok_pos.shape == (1, 8 * 64, 64)
    assert m.q_pos.shape == (1, 64, 64)
    x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    assert m(x).shape == (2, 8, 8, 8)
    preds = m.rollout(x[:, :, :8])
    assert preds.shape == (2, 8, 8, 8)
    assert set(preds.unique().tolist()) <= {0, 1}


# ---------------------------------------------------- overfit smoke (3 models)

@pytest.mark.slow
@pytest.mark.parametrize("kind", ["art_vanilla", "art_twohop", "arcnn"])
def test_baseline_overfit_smoke(kind):
    torch.manual_seed(0)
    if kind == "art_vanilla":
        model = ArtVanilla(grid=8, dmodel=64)
    elif kind == "art_twohop":
        model = ArtTwoHop(grid=8)
    else:
        model = ARCNN(grid=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    stream = StreamDataset(42, half="train", length=200 * 32,
                           rule_override=R.GOL_RULE)
    model.train()
    losses = []
    for step in range(1, 201):
        idx = list(range((step - 1) * 32, step * 32))
        frames = stream.get_batch(idx)["frames"].float()
        x = 2 * frames - 1
        loss = lossf(model(x.unsqueeze(1)), frames[:, 8:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"{kind}: {losses[0]} -> {losses[-1]}"


# ------------------------------------------------------- diffusion_vanilla

def test_diffusion_vanilla_smoke():
    torch.manual_seed(0)
    m = DiffusionVanilla(grid=8)
    x = torch.randn(4, 1, 16, 8, 8).tanh()
    loss, acc = m.training_loss(x)
    assert torch.isfinite(loss) and 0.0 <= acc <= 1.0
    loss.backward()
    preds = m.rollout(x[:, :, :8])
    assert preds.shape == (4, 8, 8, 8)
    assert set(preds.unique().tolist()) <= {0, 1}


def test_diffusion_vanilla_marginal_mc():
    """Closed-form noising marginal matches the recurrent chain (MC)."""
    rng = np.random.default_rng(0)
    N, t = 100_000, 7
    x0 = np.ones(N)
    x = x0.copy()
    for _ in range(t):
        x = RETAIN * x + SIGMA1 * rng.standard_normal(N)
    cf = RETAIN ** t * x0 + math.sqrt(1 - RETAIN ** (2 * t)) * rng.standard_normal(N)
    for xs in (x, cf):
        assert abs(xs.mean() - RETAIN ** t) < 0.02
        assert abs(xs.std() - math.sqrt(1 - RETAIN ** (2 * t))) < 0.02
