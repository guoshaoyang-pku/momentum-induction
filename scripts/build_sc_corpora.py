"""Build self-consistent (rejection-sampled) ZS corpora — protocol note
2026-08-29: a trajectory may leave table slots uncovered, but every slot the
continuation QUERIES must be covered by prefix evidence (L3); for L4 the
registered prior slot (e9, p=1.0) is exempt (resolvable by the prior — that is
the arbitration being tested), all other queried slots must be covered.

Additive: writes NEW npz files; frozen corpora are never touched.
Run: PYTHONPATH=scripts python3 scripts/build_sc_corpora.py
"""
import json
import os

import numpy as np

from cawm import rules as R
from cawm.data import PREFIX_LEN, sample_trajectory
from check_zs_protocol import frame_slots, slot_presence

N, GRID, STREAM = 2048, 8, 44


def violates(fr, cov, exempt=()):
    q = frame_slots(fr[PREFIX_LEN - 1:15])
    pres = slot_presence(q[None], 1)[0]
    return bool((pres & ~cov)[[e for e in range(R.N_ENTRIES)
                               if e not in exempt]].any())


def build(path, half, l4=None, exempt=()):
    pool = R.rule_split()[0] if half == "train" else R.rule_split()[1]
    rules, frames, covs = [], [], []
    i = 0
    while len(rules) < N:
        rule, fr = sample_trajectory(i, STREAM, pool, grid=GRID, l4=l4)
        i += 1
        from cawm.data import prefix_coverage
        cov = prefix_coverage(fr)
        if violates(fr, cov, exempt):
            continue
        rules.append(rule)
        frames.append(fr)
        covs.append(cov)
    frames = np.stack(frames)
    rules = np.array(rules, dtype=np.int64)
    covs = np.stack(covs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, frames=frames, rules=rules, covs=covs)
    import hashlib
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {
        "n": N, "master_seed": STREAM, "half": half, "grid": GRID,
        "l4": list(l4) if l4 else None, "full_coverage_only": False,
        "rejection": "context self-consistency: output-queried slots must be "
                     "prefix-covered" + (f" (exempt prior slots {list(exempt)})"
                                         if exempt else ""),
        "draws": int(i), "sha256": sha,
        "protocol": "post-submission regenerated (seed=42 split), hash-split",
    }
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{path}: n={N} draws={i} sha={sha[:12]}")


if __name__ == "__main__":
    build("data/eval_corpora/l3_zs_sc_g8.npz", "zs")
    build("data/eval_corpora/l4_zs_sc_e9_p1.0_g8.npz", "zs",
          l4=(9, 1, 1.0), exempt=(9,))
