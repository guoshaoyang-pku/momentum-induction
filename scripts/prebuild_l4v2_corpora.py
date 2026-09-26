"""Prebuild the L4-v2 pinned corpora once, before launching parallel arms
(get_corpus caches by name; concurrent first-builds would race)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cawm.train import get_corpus

for cfg in ("l4a", "l4b"):
    for name, seed, half in ((f"{cfg}_zs_sc_g8", 44, "zs"),
                             (f"{cfg}_val_sc_g8", 43, "train")):
        c = get_corpus("data/eval_corpora", name, n=2048, master_seed=seed,
                       half=half, grid=8, self_consistent=True, l4v2=cfg)
        print(name, c["frames"].shape)
