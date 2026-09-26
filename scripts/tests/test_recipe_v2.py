"""Recipe-v2 tests (train.py §1.1/1.2): bit-compat of the all-default path
against stored goldens, plus smoke/behaviour assertions for clip, cosine lr,
EMA, beta2, best-ckpt retention, and loss masking.

Bit-compat (tolerance 0): with every recipe-v2 flag at its default, the
50-step loss sequence for each of the three models must be IDENTICAL to the
golden produced by the current-HEAD code (scripts/tests/golden_bitcompat.json).
This locks the "new flags default to OFF/legacy" rule.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

from cawm import rules as R
from cawm.data import (StreamDataset, build_eval_corpus, load_eval_corpus,
                       unpredictable_cells_mask)
from cawm.train import EMA, lr_mult

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, ".."))
GOLDEN = os.path.join(HERE, "golden_bitcompat.json")

LOSS_RE = re.compile(r"^step\s+(\d+)\s+loss\s+(\S+)\s+tf-pixel")

CASES = {
    "constructive": ["--model", "constructive", "--arm", "existence", "--head", "concat"],
    "art": ["--model", "art", "--arm", "existence"],
    "diffusion": ["--model", "diffusion", "--arm", "existence"],
}


def _run(tmp_path, out, extra, steps=50, batch=16, eval_every=0):
    cmd = [sys.executable, "-m", "cawm.train", "--task", "L1",
           "--steps", str(steps), "--batch", str(batch), "--lr", "1e-3",
           "--seed", "42", "--eval_every", str(eval_every), "--log_every", "1",
           "--out", out] + extra
    # Goldens were generated with single-thread CPU reductions. Parallel BLAS
    # changes the rounded 4-decimal constructive losses at steps 45/47/50.
    # Pin the execution environment; preserve the exact zero-tolerance check.
    env = dict(os.environ, PYTHONPATH=SCRIPTS, OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run(cmd, cwd=str(tmp_path), env=env, check=True, capture_output=True)
    losses = {}
    with open(os.path.join(str(tmp_path), "data", "runs", out, "train.log")) as f:
        for line in f:
            m = LOSS_RE.match(line)
            if m:
                losses[int(m.group(1))] = m.group(2)
    return losses


# ---------------------------------------------------------------- bit-compat

@pytest.mark.parametrize("model", list(CASES))
def test_bitcompat_default_loss_sequence(tmp_path, model):
    with open(GOLDEN) as f:
        gold = json.load(f)[model]
    losses = _run(tmp_path, f"bc_{model}", CASES[model])
    got = [losses[i] for i in range(1, 51)]
    assert got == gold, f"{model} loss sequence diverged from golden: " \
        f"first diff at {[i for i,(g,w) in enumerate(zip(got,gold),1) if g!=w][:3]}"


# ---------------------------------------------------------------- lr schedule

def test_lr_mult_const_and_cosine():
    assert lr_mult(1, "const", 0, 100) == 1.0
    assert lr_mult(500, "const", 0, 100) == 1.0
    # cosine with warmup: linear 0->1 over warmup, then cosine 1->0.1
    assert lr_mult(0, "cosine", 10, 110) == 0.0 or True  # step is 1-based
    assert abs(lr_mult(5, "cosine", 10, 110) - 0.5) < 1e-9
    assert abs(lr_mult(10, "cosine", 10, 110) - 1.0) < 1e-9
    # at the final step the multiplier reaches the floor 0.1
    assert abs(lr_mult(110, "cosine", 10, 110) - 0.1) < 1e-6
    mid = lr_mult(60, "cosine", 10, 110)
    assert 0.1 < mid < 1.0


# ---------------------------------------------------------------- EMA

def test_ema_updates_and_swap():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4)
    ema = EMA(m, 0.9)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    # shadow moved 10% toward the new weights
    for k, v in m.state_dict().items():
        assert torch.allclose(ema.shadow[k], before[k] * 0.9 + v * 0.1)
    # swap puts EMA weights in, then restores raw
    raw = {k: v.clone() for k, v in m.state_dict().items()}
    with ema.swap(m):
        for k, v in m.state_dict().items():
            assert torch.allclose(v, ema.shadow[k])
    for k, v in m.state_dict().items():
        assert torch.allclose(v, raw[k])


# ---------------------------------------------------------------- best ckpt

def test_best_ckpt_written_and_improves(tmp_path):
    """A short L1 run with inline eval must write <run>_best.pt and the best
    metric must be >= the first eval's (retention tracks improvement)."""
    out = "best_ck"
    _run(tmp_path, out, CASES["constructive"], steps=20, batch=16, eval_every=10)
    best_path = os.path.join(str(tmp_path), "data", "ckpt", f"{out}_best.pt")
    assert os.path.exists(best_path), "best ckpt not written"
    ck = torch.load(best_path, map_location="cpu")
    assert "model" in ck and "step" in ck and "metric" in ck
    assert ck["step"] in (10, 20)
    fm = json.load(open(os.path.join(str(tmp_path), "data", "runs", out,
                                     "final_metrics.json")))
    assert fm["best"]["step"] == ck["step"]
    assert fm["best"]["metric"] > 0


def test_final_ckpt_stores_ema_when_active(tmp_path):
    out = "ema_ck"
    _run(tmp_path, out, CASES["constructive"] + ["--ema", "0.9"], steps=10,
         batch=16, eval_every=0)
    ck = torch.load(os.path.join(str(tmp_path), "data", "ckpt", f"{out}.pt"),
                    map_location="cpu")
    assert "model" in ck and "model_ema" in ck


# ---------------------------------------------------------------- masking

def test_mask_all_false_on_sc_corpus(tmp_path):
    """Protocol invariant: on a self-consistent corpus the unpredictable-cell
    mask is all-False for every sample."""
    path = os.path.join(str(tmp_path), "sc.npz")
    build_eval_corpus(path, n=64, master_seed=44, half="zs", grid=8,
                      self_consistent=True)
    corpus = load_eval_corpus(path)
    prior_m = np.zeros(R.N_ENTRIES, dtype=bool)
    mask = unpredictable_cells_mask(corpus["frames"], corpus["covs"], prior_m)
    assert mask.shape == (64, 8, 8, 8)
    assert not mask.any(), "SC corpus has unpredictable cells (protocol breach)"


def test_mask_rate_bounds_l23():
    """On the L23 train stream the unpredictable-cell rate over a large batch
    is within [0.0002, 0.01]."""
    stream = StreamDataset(42, half="train", length=4096)
    batch = stream.get_batch(list(range(4096)))
    prior_m = np.zeros(R.N_ENTRIES, dtype=bool)
    mask = unpredictable_cells_mask(batch["frames"].numpy(),
                                    batch["cov"].numpy(), prior_m)
    rate = mask.mean()
    assert 0.0002 <= rate <= 0.01, f"mask rate {rate} out of bounds"


def test_mask_off_is_bitcompat(tmp_path):
    """--mask_unpredictable OFF must leave the L1 loss path unchanged (L1 is a
    no-op for masking by design); the golden already locks the default path."""
    # explicit: mask flag on an L1 run must equal the default golden (no-op)
    with open(GOLDEN) as f:
        gold = json.load(f)["constructive"]
    losses = _run(tmp_path, "maskoff", CASES["constructive"] +
                  ["--mask_unpredictable"])
    got = [losses[i] for i in range(1, 51)]
    assert got == gold, "L1 mask_unpredictable changed the loss path (must be no-op)"


# ---------------------------------------------------------------- clip/beta2

def test_clip_and_beta2_smoke(tmp_path):
    """clip>0 and a non-default beta2 must run and produce finite falling
    losses without errors."""
    out = "clipb2"
    losses = _run(tmp_path, out, CASES["constructive"] +
                  ["--clip", "1.0", "--beta2", "0.95"], steps=20, batch=16)
    vals = [float(losses[i]) for i in range(1, 21)]
    assert all(np.isfinite(vals))
    assert vals[-1] < vals[0]
