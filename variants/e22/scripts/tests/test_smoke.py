"""Smoke tests: simulator correctness, stream determinism, architecture specs,
analytic detector exactness, and a small L1 overfit run."""

import numpy as np
import pytest
import torch

from cawm import rules as R
from cawm.data import StreamDataset, prefix_coverage, sample_trajectory
from cawm.models import ConstructiveCNN
from cawm.simulate import ca_step, neighbor_count, rule_tables, simulate, to_pm1


def _blinker():
    x = np.zeros((1, 8, 8), dtype=np.uint8)
    x[0, 4, 3:6] = 1
    return x


def test_gol_blinker():
    frames = simulate(np.array([R.GOL_RULE]), _blinker(), 2)
    assert frames[0, 0].sum() == 3
    v = frames[0, 1]
    assert set(zip(*np.nonzero(v))) == {(3, 4), (4, 4), (5, 4)}  # vertical
    assert (frames[0, 2] == frames[0, 0]).all()  # period 2


def test_trivial_rules():
    x = _blinker()
    dead = simulate(np.array([0]), x, 1)      # all entries 0 -> everything dies
    assert dead[0, 1].sum() == 0
    alive = simulate(np.array([2 ** 18 - 1]), x, 1)  # all entries 1 -> all black
    assert alive[0, 1].sum() == 64


def test_rule_table_convention():
    # rule with a single 1-bit at entry (s=1, n=3) -> only live cells with 3
    # live neighbors survive/birth nothing
    entry = 9 * 1 + 3
    rule = 1 << entry
    x = np.zeros((1, 8, 8), dtype=np.uint8)
    x[0, 4, 3:6] = 1  # blinker: center alive with 2 alive neighbors
    nxt = ca_step(x, rule_tables(np.array([rule])))
    # center: s=1, n=2 -> entry 11 -> 0; side cells: s=1, n=1 -> 0;
    # cells above/below center: s=0, n=3 -> entry 3 -> 0
    assert nxt.sum() == 0
    x2 = np.zeros((1, 8, 8), dtype=np.uint8)
    x2[0, 4, 4] = 1
    x2[0, 3, 3] = 1
    x2[0, 5, 5] = 1
    x2[0, 3, 5] = 1
    # (4,4): alive, neighbors = (3,3),(5,5),(3,5) -> n=3 -> survives
    nxt2 = ca_step(x2, rule_tables(np.array([rule])))
    assert nxt2[0, 4, 4] == 1 and nxt2.sum() == 1


def test_neighbor_count_wrap():
    x = np.zeros((1, 8, 8), dtype=np.uint8)
    x[0, 0, 0] = 1
    n = neighbor_count(x)
    assert n[0, 7, 7] == 1 and n[0, 0, 1] == 1 and n[0, 1, 0] == 1 and n[0, 1, 1] == 1
    assert n[0, 0, 0] == 0 and n.sum() == 8


def test_split_unbiased_and_disjoint():
    tr, zs = R.rule_split()
    assert len(tr) == len(zs) == 131072
    assert not np.intersect1d(tr, zs).size
    assert np.array_equal(np.sort(np.concatenate([tr, zs])), np.arange(2 ** 18))
    tr2, _ = R.rule_split()
    assert np.array_equal(tr, tr2)  # cached, deterministic


def test_stream_determinism():
    pool = R.rule_split()[0]
    r1, f1 = sample_trajectory(7, 42, pool)
    r2, f2 = sample_trajectory(7, 42, pool)
    assert r1 == r2 and np.array_equal(f1, f2)
    r3, f3 = sample_trajectory(8, 42, pool)
    assert r3 != r1 or not np.array_equal(f1, f3)


def test_chunked_loader_bit_identity():
    """DataLoader workers (any count) must yield bit-identical batches vs sync."""
    from cawm.data import ChunkedStreamDataset
    from torch.utils.data import DataLoader

    n_chunks, cs = 6, 32
    stream = StreamDataset(42, half="train", length=n_chunks * cs)
    chunked = ChunkedStreamDataset(42, half="train", n_chunks=n_chunks, chunk_size=cs)
    loader = DataLoader(chunked, batch_size=None, num_workers=2)
    got = list(loader)
    assert len(got) == n_chunks
    for c, batch in enumerate(got):
        ref = stream.get_batch(list(range(c * cs, (c + 1) * cs)))
        assert torch.equal(batch["frames"], ref["frames"])
        assert torch.equal(batch["rule"], ref["rule"])
        assert torch.equal(batch["cov"], ref["cov"])


def test_model_init_isolated_from_loader_spawn():
    """Author decision 2026-08-29: model seed and data seed are consumed
    separately (both 42 by default). build_model's isolated re-seed must make
    model init invariant to DataLoader worker spawning (which draws its base
    seed from the global torch RNG), regardless of ordering."""
    from cawm.data import ChunkedStreamDataset
    from cawm.train import build_model
    from torch.utils.data import DataLoader

    m_before = build_model(7, grid=8, arm="emergence", head="concat")

    ds = ChunkedStreamDataset(42, n_chunks=2, chunk_size=8)
    it = iter(DataLoader(ds, batch_size=None, num_workers=2))
    next(it)  # spawn workers + consume the base-seed draw from the global RNG

    m_after = build_model(7, grid=8, arm="emergence", head="concat")
    assert all(torch.equal(a, b) for a, b in zip(m_before.parameters(),
                                                 m_after.parameters()))


def test_cov_bins_metric():
    from cawm.eval import metrics_cov_bins
    corpus = {
        "frames": np.random.default_rng(0).integers(0, 2, (16, 16, 8, 8)).astype(np.uint8),
        "covs": np.zeros((16, 18), dtype=bool),
    }
    corpus["covs"][:8, :18] = True   # 8 full-coverage, 8 zero-coverage
    preds = corpus["frames"][:, 8:].copy()  # perfect predictions
    m = metrics_cov_bins(preds, corpus)
    assert m[18]["n"] == 8 and m[18]["seq_acc"] == 1.0
    assert m[0]["n"] == 8 and m[0]["seq_acc"] == 1.0
    assert m["mean_cov"] == 9.0


def test_pm1_roundtrip():
    x = np.array([[0, 1]], dtype=np.uint8)
    assert np.array_equal(to_pm1(x), np.array([[-1.0, 1.0]]))


def test_param_counts():
    m = ConstructiveCNN(grid=8, arm="emergence", head="bilinear")
    assert m.trainable_param_count() == 4686  # docs/SETTINGS.md §6
    m16 = ConstructiveCNN(grid=16, arm="emergence", head="bilinear")
    assert m16.trainable_param_count() == 5046
    e = ConstructiveCNN(grid=8, arm="existence", head="bilinear")
    assert e.trainable_param_count() == 2053  # head only
    assert e.total_param_count() == 2053 + 57 + 96 + 900 + 20 + 66 + 414
    mc = ConstructiveCNN(grid=8, arm="emergence", head="concat")
    assert mc.trainable_param_count() == 17994
    mc16 = ConstructiveCNN(grid=16, arm="emergence", head="concat")
    assert mc16.trainable_param_count() == 18354
    ec = ConstructiveCNN(grid=8, arm="existence", head="concat")
    assert ec.trainable_param_count() == 15361
    assert ec.total_param_count() == 15361 + 57 + 96 + 900 + 20 + 66 + 414


def _place_pattern(grid=8):
    """Coordinates: center (4,4) plus its 8 neighbors in fixed order."""
    c = (4, 4)
    nbrs = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3)]
    return c, nbrs


def test_existence_detectors_A():
    m = ConstructiveCNN(grid=8, arm="existence")
    c, nbrs = _place_pattern()
    for s in (0, 1):
        for n in range(9):
            for o in (0, 1):
                f0 = np.full((8, 8), -1.0, dtype=np.float32)
                if s:
                    f0[c] = 1.0
                for (y, x) in nbrs[:n]:
                    f0[y, x] = 1.0
                f1 = np.full((8, 8), -1.0, dtype=np.float32)
                if o:
                    f1[c] = 1.0
                inp = torch.from_numpy(np.stack([f0, f1])[None, None])  # (1,1,2,8,8)
                with torch.no_grad():
                    h = m.feat_a(torch.nn.functional.pad(inp, (1, 1, 1, 1, 0, 0), mode="circular"))
                    h = torch.relu(m.split_a(h))
                    h = torch.relu(m.conjoin_a(h))
                got = h[0, :, 0, c[0], c[1]].numpy()
                want = np.zeros(36)
                want[2 * (9 * s + n) + o] = 1.0
                assert np.allclose(got, want), f"slot (s={s},n={n},o={o}): {got}"


def test_existence_detectors_B():
    m = ConstructiveCNN(grid=8, arm="existence")
    c, nbrs = _place_pattern()
    for s in (0, 1):
        for n in range(9):
            f0 = np.full((8, 8), -1.0, dtype=np.float32)
            if s:
                f0[c] = 1.0
            for (y, x) in nbrs[:n]:
                f0[y, x] = 1.0
            inp = torch.from_numpy(f0[None, None])
            with torch.no_grad():
                h = m.feat_b(torch.nn.functional.pad(inp, (1, 1, 1, 1), mode="circular"))
                h = torch.relu(m.split_b(h))
                h = torch.relu(m.conjoin_b(h))
            got = h[0, :, c[0], c[1]].numpy()
            want = np.zeros(18)
            want[9 * s + n] = 1.0
            assert np.allclose(got, want), f"code (s={s},n={n}): {got}"


def test_forward_shapes():
    for head in ("concat", "bilinear"):
        m = ConstructiveCNN(grid=8, arm="existence", head=head)
        x = torch.randn(2, 1, 16, 8, 8)
        assert m(x).shape == (2, 8, 8, 8)
        cond = m.forward_a(x[:, :, :8])
        assert cond.shape == (2, 36)
        preds = m.rollout(x[:, :, :8])
        assert preds.shape == (2, 8, 8, 8)
        assert set(preds.unique().tolist()) <= {0, 1}


def test_prefix_coverage():
    # GoL still life (block): coverage should include (1,3),(1,4),(0,4)... a
    # 2x2 block: each cell has 3 live neighbors, all alive -> slot (1,3) only,
    # plus dead contexts around it.
    fr = np.zeros((16, 8, 8), dtype=np.uint8)
    fr[:, 4:6, 4:6] = 1
    cov = prefix_coverage(fr)
    assert cov[9 * 1 + 3]  # live cell with 3 live neighbors
    assert cov.sum() >= 1
    assert cov.shape == (18,)


@pytest.mark.slow
def test_l1_existence_overfit(tmp_path):
    """Existence arm on a single rule must reach high SeqAcc quickly."""
    from cawm.data import build_eval_corpus, load_eval_corpus
    import os
    path = os.path.join(str(tmp_path), "l1.npz")
    build_eval_corpus(path, n=256, master_seed=43, half="train",
                      rule_override=R.GOL_RULE)
    corpus = load_eval_corpus(path)

    torch.manual_seed(0)
    model = ConstructiveCNN(grid=8, arm="existence")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    stream = StreamDataset(42, half="train", length=200 * 64, rule_override=R.GOL_RULE)
    model.train()
    for step in range(1, 201):
        batch = [stream[i] for i in range((step - 1) * 64, step * 64)]
        frames = torch.stack([b["frames"] for b in batch]).float()
        x = 2 * frames - 1
        logits = model(x.unsqueeze(1))
        loss = lossf(logits, frames[:, 8:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    from cawm.eval import metrics, rollout_corpus
    preds = rollout_corpus(model, corpus)
    m = metrics(preds, corpus)
    assert m["seq_acc"] >= 0.9, m


def test_resume_bit_identity(tmp_path):
    """--resume must continue a run bit-identically: fresh 24-step run vs
    12-step run resumed to 24 produce the same loss line sequence."""
    import os
    import subprocess
    import sys

    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = dict(os.environ, PYTHONPATH=scripts_dir)

    def run(out, steps, extra):
        cmd = [sys.executable, "-m", "cawm.train", "--task", "L1",
               "--arm", "existence", "--steps", str(steps), "--batch", "16",
               "--lr", "1e-3", "--seed", "42", "--eval_every", "24",
               "--log_every", "1", "--out", out] + extra
        subprocess.run(cmd, cwd=str(tmp_path), env=env, check=True,
                       capture_output=True)

    run("rfull", 24, [])
    run("rpart", 12, ["--save_every", "6"])
    run("rpart", 24, ["--resume"])  # continues 13..24 from the step-12 ckpt

    def losses(logfile):
        out = []
        for line in open(logfile):
            if line.startswith("step "):
                p = line.split()
                out.append((int(p[1]), p[3]))
        return out

    a = losses(os.path.join(str(tmp_path), "data", "runs", "rfull", "train.log"))
    b = losses(os.path.join(str(tmp_path), "data", "runs", "rpart", "train.log"))
    # rpart log holds 12 fresh steps then 12 resumed steps; dedupe the overlap
    seen, merged = set(), []
    for step, loss in b:
        if step not in seen:
            seen.add(step)
            merged.append((step, loss))
    assert [s for s, _ in merged] == list(range(1, 25))
    assert a == merged, (a[:5], merged[:5])
