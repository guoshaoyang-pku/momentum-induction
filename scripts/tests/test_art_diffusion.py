"""ART induction-head and diffusion tests (docs/SETTINGS.md §6.5 + ART spec).

Covers: VP schedule correctness, closed-form noising vs the recurrent chain,
the analytic posterior step (Monte Carlo), batch-stratified t, chain level
sub-sampling, ART token/attention exactness (existence arm), parameter-count
locks, forward/rollout shapes, and a small L1 overfit for the diffusion
denoiser.
"""

import math

import numpy as np
import pytest
import torch

from cawm.models import ArtInduction, ArtKVShift, DiffusionModel
from cawm.models.diffusion import (RETAIN, SIGMA1, T_STEPS, FEAT_CH, alpha_t,
                                   chain_levels, sigma_t, stratified_t)


# ---------------------------------------------------------------- schedule

def test_vp_schedule_values():
    assert abs(alpha_t(0) - 1.0) < 1e-12
    assert abs(alpha_t(1) - 0.9) < 1e-12
    assert abs(SIGMA1 - math.sqrt(0.19)) < 1e-12
    assert abs(sigma_t(1) - SIGMA1) < 1e-12
    # variance preservation at every level: a_t^2 + sigma_t^2 == 1
    t = torch.arange(0, T_STEPS, dtype=torch.float64)
    av, sv = alpha_t(t) ** 2, sigma_t(t) ** 2
    assert torch.allclose(av + sv, torch.ones(T_STEPS, dtype=torch.float64))
    # a_44 < 0.01 (SNR -40 dB) and sigma_49 ~ 1
    assert alpha_t(44) < 0.01
    assert abs(sigma_t(49) - 1.0) < 1e-3
    t44 = torch.tensor(44.0)
    assert float(alpha_t(t44)) < 0.01                      # tensor branch too


def test_closed_form_matches_chain():
    rng = np.random.default_rng(0)
    N, t = 200_000, 7
    x0 = np.ones(N)                                        # condition on x0=+1
    x = x0.copy()
    for _ in range(t):
        x = RETAIN * x + SIGMA1 * rng.standard_normal(N)
    cf = RETAIN ** t * x0 + math.sqrt(1 - RETAIN ** (2 * t)) * rng.standard_normal(N)
    for xs in (x, cf):
        assert abs(xs.mean() - RETAIN ** t) < 0.01
        assert abs(xs.std() - math.sqrt(1 - RETAIN ** (2 * t))) < 0.01


def test_posterior_step_monte_carlo():
    """Frozen posterior for one ancestral step, checked at t=2 -> t=1."""
    rng = np.random.default_rng(1)
    N = 400_000
    x0 = np.ones(N)
    s1, s2 = math.sqrt(1 - RETAIN ** 2), math.sqrt(1 - RETAIN ** 4)
    eta, e2 = rng.standard_normal(N), rng.standard_normal(N)
    x1 = RETAIN * x0 + s1 * eta
    x2 = RETAIN * x1 + s1 * e2
    mu = (RETAIN * SIGMA1 ** 2 / s2 ** 2) * x0 + (RETAIN * s1 ** 2 / s2 ** 2) * x2
    var = SIGMA1 ** 2 * s1 ** 2 / s2 ** 2
    resid = x1 - mu
    assert abs(resid.mean()) < 0.01
    assert abs(resid.var() - var) < 0.01


def test_stratified_t_coverage():
    torch.manual_seed(0)
    t = stratified_t(128)
    counts = torch.bincount(t, minlength=T_STEPS)
    assert counts.min() >= 128 // T_STEPS and counts.sum() == 128
    t2 = stratified_t(128)
    assert not torch.equal(t, t2)          # shuffled; consumes the RNG stream
    t3 = stratified_t(7)                   # batch < T: no repeats guaranteed
    assert torch.unique(t3).numel() == 7


def test_chain_levels():
    assert chain_levels(49) == list(range(49, 0, -1))
    lv = chain_levels(4)
    assert lv == sorted(lv, reverse=True) and lv[0] == 49 and lv[-1] == 1
    assert len(lv) == 4
    assert chain_levels(1) == [49]
    full = chain_levels(50)
    assert all(full[i] - full[i + 1] == 1 for i in range(len(full) - 1))


# ---------------------------------------------------------------- ART

def test_art_param_counts():
    e = ArtInduction(grid=8, arm="existence")
    assert e.trainable_param_count() == 913     # head only: 36*24+24 + 24+1
    m = ArtInduction(grid=8, arm="emergence")
    # A stack 57+108+1008, B stack 20+54+342, Wk/Wv 648+648, head 913
    assert m.trainable_param_count() == 1173 + 416 + 648 + 648 + 913


def _art_prefix():
    """8-frame prefix isolating six (s=1,n=2) contexts with outcomes
    {1,0,0,0,0,0} in the last transition, plus a query cell (0,0) whose
    frame-7 context is (s=1,n=2)."""
    f = np.full((8, 8, 8), -1.0, dtype=np.float32)
    for (y, x) in [(2, 2), (1, 2), (2, 1), (5, 5), (4, 5), (5, 4)]:
        f[6, y, x] = 1.0                                  # frame 6 live cells
    for (y, x) in [(2, 2), (0, 0), (0, 1), (1, 0)]:       # frame 7 live cells
        f[7, y, x] = 1.0
    return torch.from_numpy(f)[None, None]                # (1,1,8,8,8)


def test_art_tokens_onehot():
    m = ArtInduction(grid=8, arm="existence")
    tok = m._tokens(_art_prefix())
    assert tok.shape == (1, 7 * 64, 36)
    # transition (6,7), cell (2,2): context (1,2) outcome 1 -> channel 23
    got = tok[0, 6 * 64 + 2 * 8 + 2].numpy()
    want = np.zeros(36)
    want[2 * 11 + 1] = 1.0
    assert np.allclose(got, want, atol=1e-5)
    # a white-background transition yields the (0,0) context slot
    got0 = tok[0, 0 * 64 + 7 * 8 + 7].numpy()
    want0 = np.zeros(36)
    want0[0] = 1.0                                        # channel 2*(9*0+0)+0
    assert np.allclose(got0, want0, atol=1e-5)


def test_art_attention_exact():
    m = ArtInduction(grid=8, arm="existence")
    prefix = _art_prefix()
    tok = m._tokens(prefix)
    q = m._codes(prefix[:, :, 7])                         # (1,18,8,8)
    q = q.reshape(1, 18, 64).permute(0, 2, 1)             # (1,64,18)
    out = m._attend(tok, q)
    got = out[0, 0]                                       # cell (0,0)
    assert abs(got[11].item() - 1.0 / 6.0) < 1e-4         # mean of {1,0,0,0,0,0}
    mask = torch.ones(18, dtype=torch.bool)
    mask[11] = False
    assert got[mask].abs().max() < 1e-3
    # a cell with an unseen context (7,7) is dead in frame 7 -> context (0,0)
    # which IS seen in transitions 0..5 (all white) with outcome 0
    q2 = m._codes(prefix[:, :, 7])
    out2 = m._attend(tok, q2.reshape(1, 18, 64).permute(0, 2, 1))
    assert abs(out2[0, 7 * 8 + 7, 0].item()) < 1e-4       # white stays white


def test_art_forward_and_rollout_shapes():
    for arm in ("existence", "emergence"):
        m = ArtInduction(grid=8, arm=arm)
        x = torch.randn(2, 1, 16, 8, 8)
        assert m(x).shape == (2, 8, 8, 8)
        preds = m.rollout(x[:, :, :8])
        assert preds.shape == (2, 8, 8, 8)
        assert set(preds.unique().tolist()) <= {0, 1}


@pytest.mark.slow
def test_art_l1_existence_overfit(tmp_path):
    """Existence ART on a single rule must reach high SeqAcc quickly."""
    import os

    from cawm import rules as R
    from cawm.data import StreamDataset, build_eval_corpus, load_eval_corpus
    from cawm.eval import metrics, rollout_corpus

    path = os.path.join(str(tmp_path), "l1.npz")
    build_eval_corpus(path, n=256, master_seed=43, half="train",
                      rule_override=R.GOL_RULE)
    corpus = load_eval_corpus(path)

    torch.manual_seed(0)
    model = ArtInduction(grid=8, arm="existence")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    stream = StreamDataset(42, half="train", length=300 * 64, rule_override=R.GOL_RULE)
    model.train()
    for step in range(1, 301):
        idx = list(range((step - 1) * 64, step * 64))
        frames = stream.get_batch(idx)["frames"].float()
        x = 2 * frames - 1
        loss = lossf(model(x.unsqueeze(1)), frames[:, 8:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds = rollout_corpus(model, corpus)
    m = metrics(preds, corpus)
    assert m["seq_acc"] >= 0.85, m


# ------------------------------------------------------- ART KV-shift (E4.7)

def test_kvshift_param_counts():
    e = ArtKVShift(grid=8, arm="existence")
    assert e.trainable_param_count() == 913     # head only, same as ArtInduction
    m = ArtKVShift(grid=8, arm="emergence")
    # shared detector 20+54+342, Wk/Wv 648+648, head 913
    assert m.trainable_param_count() == 416 + 648 + 648 + 913


def test_kvshift_tokens_match_pairing_conv():
    """The E4.7 invariant: analytic KV-shift tokens are bit-equal to the
    pairing conv's tokens — the rung isolates pairing-as-alignment from
    pairing-as-convolution."""
    torch.manual_seed(0)
    ref = ArtInduction(grid=8, arm="existence")
    kv = ArtKVShift(grid=8, arm="existence")
    for _ in range(4):
        prefix = (torch.randint(0, 2, (2, 1, 8, 8, 8)).float() * 2 - 1)
        a = ref._tokens(prefix)
        b = kv._tokens(prefix)
        assert a.shape == b.shape == (2, 7 * 64, 36)
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_kvshift_rollout_matches_art():
    """With the same head, rollout is identical to the pairing-conv ART."""
    torch.manual_seed(0)
    ref = ArtInduction(grid=8, arm="existence")
    kv = ArtKVShift(grid=8, arm="existence")
    kv.head.load_state_dict(ref.head.state_dict())
    prefix = (torch.randint(0, 2, (3, 1, 8, 8, 8)).float() * 2 - 1)
    assert torch.equal(ref.rollout(prefix), kv.rollout(prefix))


def test_kvshift_forward_and_rollout_shapes():
    for arm in ("existence", "emergence"):
        m = ArtKVShift(grid=8, arm=arm)
        x = torch.randn(2, 1, 16, 8, 8)
        assert m(x).shape == (2, 8, 8, 8)
        preds = m.rollout(x[:, :, :8])
        assert preds.shape == (2, 8, 8, 8)
        assert set(preds.unique().tolist()) <= {0, 1}


# ---------------------------------------------------------------- diffusion

def test_diffusion_param_counts():
    e = DiffusionModel(grid=8, arm="existence")
    # v3 cascade head: enc 608+18464; lookup (18+36)->64->1 = 3585;
    # corr (32+38)->64->1 = 4609
    assert e.trainable_param_count() == 608 + 18464 + 3585 + 4609
    m = DiffusionModel(grid=8, arm="emergence")
    # + A stack 57+108+1008, gate 36, pyramid 2*360, temporal 252+36, B 20+54+342
    assert m.trainable_param_count() == 27266 + 1173 + 36 + 720 + 288 + 416


def test_diffusion_training_loss_shapes_and_finiteness():
    for arm in ("existence", "emergence"):
        torch.manual_seed(0)
        m = DiffusionModel(grid=8, arm=arm)
        x = torch.randn(4, 1, 16, 8, 8).tanh()            # pseudo ±1 frames
        loss, acc = m.training_loss(x)
        assert torch.isfinite(loss) and 0.0 <= acc <= 1.0
        loss.backward()


def test_diffusion_rollout_shapes():
    torch.manual_seed(0)
    for arm, nfe in (("existence", 49), ("emergence", 5)):
        m = DiffusionModel(grid=8, arm=arm).eval()
        prefix = torch.randn(2, 1, 8, 8, 8).tanh()
        preds = m.rollout(prefix, nfe=nfe)
        assert preds.shape == (2, 8, 8, 8)
        assert set(preds.unique().tolist()) <= {0, 1}


def test_diffusion_encoder_temporal_coverage():
    """Every canvas frame must receive encoder features (temporal pad)."""
    m = DiffusionModel(grid=8, arm="existence")
    xt = torch.randn(2, 1, 8, 8, 8)
    feat = m._encode(xt)
    assert feat.shape == (2, FEAT_CH, 8, 8, 8)
    assert torch.isfinite(feat).all()


@pytest.mark.slow
def test_diffusion_l1_existence_overfit(tmp_path):
    """Existence diffusion denoiser on one rule: tf-pixel accuracy rises."""
    import os

    from cawm import rules as R
    from cawm.data import StreamDataset

    torch.manual_seed(0)
    model = DiffusionModel(grid=8, arm="existence")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=2e-3)
    stream = StreamDataset(42, half="train", length=300 * 64, rule_override=R.GOL_RULE)
    model.train()
    first_acc = None
    for step in range(1, 301):
        idx = list(range((step - 1) * 64, step * 64))
        frames = stream.get_batch(idx)["frames"].float()
        loss, acc = model.training_loss((2 * frames - 1).unsqueeze(1))
        if first_acc is None:
            first_acc = acc.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert acc.item() > 0.75, (first_acc, acc.item())
    assert acc.item() > first_acc + 0.2
