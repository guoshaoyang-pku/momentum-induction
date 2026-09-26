"""E6.2/E6.3 L4-v2 constrained families: pool, predictability, corpus, metrics."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cawm import rules as R
from cawm.data import PREFIX_LEN, build_eval_corpus, queried_slots_mask
from cawm.eval import metrics_stratified_v2


def _satisfies(rule_ids, cfg):
    ok = np.ones(len(rule_ids), dtype=bool)
    for e, v in cfg["defaults"]:
        ok &= ((rule_ids >> e) & 1) == v
    for a, b in cfg["pairs"]:
        ok &= ((rule_ids >> a) & 1) != ((rule_ids >> b) & 1)
    return ok


def test_l4v2_pools_constraints_and_disjointness():
    for name, nbits in (("l4a", 4), ("l4b", 2)):
        cfg = R.L4V2_CONFIGS[name]
        tr = R.l4v2_pool(name, "train")
        zs = R.l4v2_pool(name, "zs")
        assert _satisfies(tr, cfg).all() and _satisfies(zs, cfg).all()
        # split decided on the constrained rule id: halves truly disjoint
        assert len(np.intersect1d(tr, zs)) == 0
        assert len(tr) + len(zs) == 2 ** (18 - nbits)  # whole family split
        # roughly balanced hash split
        assert abs(len(tr) - len(zs)) < 0.05 * (len(tr) + len(zs))


def test_l4v2_effective_coverage():
    covs = np.zeros((3, 18), dtype=bool)
    covs[1, 4] = True   # pair (4,13) member a covered
    covs[2, 13] = True  # partner side
    eff = R.l4v2_effective_coverage(covs, "l4a")
    for e, _ in R.L4V2_CONFIGS["l4a"]["defaults"]:
        assert eff[:, e].all()          # defaults always predictable
    assert not eff[0, 4] and not eff[0, 13]
    eff = R.l4v2_effective_coverage(covs, "l4b")
    assert eff[1, 4] and eff[1, 13]     # either member covers both
    assert eff[2, 4] and eff[2, 13]
    assert not eff[0, 4] and not eff[0, 13]
    assert not eff[0, 5]                # untouched slot stays uncovered


def _build(name, n=64, half="zs", seed=44):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.npz")
        build_eval_corpus(path, n, seed, half=half, grid=8,
                          self_consistent=True, l4v2=name)
        z = np.load(path)
        return {"frames": z["frames"], "covs": z["covs"],
                "rules": z["rules"]}


def test_l4v2_corpus_invariants():
    for name in ("l4a", "l4b"):
        c = _build(name)
        cfg = R.L4V2_CONFIGS[name]
        assert _satisfies(c["rules"].astype(np.int64), cfg).all()
        zs = R.l4v2_pool(name, "zs")
        assert np.isin(c["rules"], zs).all()
        # SC invariant: every queried slot predictable under the constraints
        eff = R.l4v2_effective_coverage(c["covs"], name)
        assert not (queried_slots_mask(c["frames"]) & ~eff).any()


def test_l4v2_strata_present_and_metric_runs():
    for name, key in (("l4a", "default_exercised_frac"),
                      ("l4b", "derived_exercised_frac")):
        c = _build(name, n=128)
        # perfect predictions -> every stratum reports 1.0
        preds = c["frames"][:, PREFIX_LEN:].copy()
        m = metrics_stratified_v2(preds, c, name)
        assert key in m and 0.0 <= m[key] <= 1.0
        for k, v in m.items():
            if k.startswith("seq_acc"):
                assert v == 1.0


# ---- E6.6 exercised pool / oversampling ----
def test_exercised_mask_matches_corpus_filter():
    """l4v2_exercised_mask reproduces the require_exercised corpus filter on
    a raw batch: exercised trajectories are SC-valid-or-not agnostic, but
    every kept trajectory in an scx corpus must be exercised."""
    import numpy as np
    from cawm.data import l4v2_exercised_mask, sample_batch, queried_slots_mask
    from cawm import rules as R
    pool = R.l4v2_pool("l4b", "zs")
    rules, frames, covs = sample_batch(range(20000), 45, pool, grid=8)
    ex = l4v2_exercised_mask(frames, covs, "l4b")
    # exercised => a pair slot is queried while uncovered with partner covered
    cfg = R.L4V2_CONFIGS["l4b"]
    q = queried_slots_mask(frames)
    manual = np.zeros(len(rules), dtype=bool)
    for a, b in cfg["pairs"]:
        manual |= (q[:, b] & ~covs[:, b] & covs[:, a]) | (q[:, a] & ~covs[:, a] & covs[:, b])
    assert np.array_equal(ex, manual)
    assert 0 < ex.sum() < 0.05 * len(rules)   # rare, but present


def test_ex_pool_mixing_is_deterministic_and_off_by_default(tmp_path):
    import numpy as np
    from cawm.data import ChunkedStreamDataset, l4v2_exercised_mask
    from cawm import rules as R
    # tiny synthetic pool: 5 exercised-looking entries (content irrelevant
    # for the addressing test; shapes must match)
    pool = R.l4v2_pool("l4b", "train")
    from cawm.data import sample_batch
    r, f, c = sample_batch(range(5), 999, pool, grid=8)
    pth = str(tmp_path / "pool.npz")
    np.savez_compressed(pth, frames=f, rules=r, covs=c)
    base = ChunkedStreamDataset(42, half="train", n_chunks=4, chunk_size=16,
                                sc_filter=True, l4v2="l4b")
    mixed = ChunkedStreamDataset(42, half="train", n_chunks=4, chunk_size=16,
                                 sc_filter=True, l4v2="l4b", ex_pool=pth, ex_frac=0.25)
    off = ChunkedStreamDataset(42, half="train", n_chunks=4, chunk_size=16,
                               sc_filter=True, l4v2="l4b", ex_pool=pth, ex_frac=0.0)
    for ch in range(4):
        b0, bm, boff = base[ch], mixed[ch], off[ch]
        assert np.array_equal(b0["frames"].numpy(), boff["frames"].numpy())
        assert bm["frames"].shape == b0["frames"].shape
        # tail (12 entries) identical to the unmixed stream; head = pool rows
        assert np.array_equal(bm["frames"][4:].numpy(), b0["frames"][4:].numpy())
        j = (ch * 4 + np.arange(4)) % 5
        assert np.array_equal(bm["frames"][:4].numpy(), f[j])
        assert np.array_equal(bm["rule"][:4].numpy(), r[j])
        assert np.array_equal(mixed[ch]["frames"].numpy(), bm["frames"].numpy())
