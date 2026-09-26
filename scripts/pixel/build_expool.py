"""Build exercised pools for pixel-lab L4 levels (vectorized, parallel).

Pools store only cell states (16x8x8 uint8 = 128B/traj) + rule ids;
rendering happens online in the stream. Usage:
  python3 -m pixel.build_expool l4b data/pools/l4b_train_expool.npz 50000 16
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

from .pool_batch import draw_exercised_pool


def _work(arg):
    cfg, half, n, seed = arg
    out = draw_exercised_pool(cfg, n, half, seed=seed, batch=8192)
    return out["states"], out["rules"], out["drawn"]


def main():
    cfg, path, n_target, procs = sys.argv[1], sys.argv[2], int(sys.argv[3]), \
        int(sys.argv[4])
    t0 = time.time()
    per = n_target // procs + 1
    with Pool(procs) as p:
        parts = p.map(_work, [(cfg, "train", per, 1000 + i)
                              for i in range(procs)])
    states = np.concatenate([s for s, _, _ in parts])[:n_target]
    rules = np.concatenate([r for _, r, _ in parts])[:n_target]
    drawn = sum(d for _, _, d in parts)
    np.savez_compressed(path, states=states, rules=rules)
    print(f"{cfg}: {len(rules)} exercised trajs from {drawn} draws "
          f"({time.time()-t0:.0f}s, rate {len(rules)/max(drawn,1):.5f}), "
          f"{len(np.unique(rules))} unique rules -> {path}")


if __name__ == "__main__":
    main()
