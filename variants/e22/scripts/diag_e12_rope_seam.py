"""E12.3 (SETTINGS S14): where does the RoPE raster decoder fail on Game of Life?

Zero training. For each art_vanilla checkpoint: teacher-forced per-cell error
map over (row, col) for the 8 predicted frames, and the rollout first-error
cells, on l1_val_gol_g8 (n=512). Seam cells: row in {0,7} or col in {0,7}
(28 of 64). In raster order their previous-frame 3x3 neighbourhood sits at
irregular relative offsets, which a purely relative encoding cannot separate
from the interior offsets. Writes data/runs/<run>/e12_rope_seam.json.

Usage (repo root, on a node):
  PYTHONPATH=scripts python3 scripts/diag_e12_rope_seam.py --device cuda \
      --runs artvan_L1_d64_perope_s42_bf16 artvan_L1_d64_pelearned_s42_bf16 \
             artvan_L1_d64_pefactorized_s42_bf16
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from cawm import rules as R
from cawm.train import build_model, code_fingerprint, get_corpus, model_kwargs


def load(run, device):
    ck = torch.load(os.path.join("data", "ckpt", f"{run}.pt"), map_location="cpu",
                    weights_only=False)
    args = ck["args"]
    m = build_model(args["seed"], model=args["model"], **model_kwargs(args))
    m.load_state_dict({**ck["model"], **ck.get("model_ema", {})})
    return m.to(device).eval(), args, int(ck.get("step", -1))


def seam_mask(g):
    s = np.zeros((g, g), dtype=bool)
    s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = True
    return s


@torch.no_grad()
def analyse(m, frames, device, batch=64):
    N, _, G, _ = frames.shape
    tf_err = np.zeros((8, G, G), dtype=np.int64)
    first_err = np.zeros((G, G), dtype=np.int64)
    first_frame = np.zeros(8, dtype=np.int64)
    exact = 0
    roll_err = np.zeros((8, G, G), dtype=np.int64)
    exact_by_world = []
    for i in range(0, N, batch):
        fr = frames[i:i + batch]
        x = torch.as_tensor(fr.astype(np.float32), device=device) * 2 - 1
        logits = m(x.unsqueeze(1))                          # (B,8,G,G), frames 8..15
        pred = (logits > 0).cpu().numpy().astype(np.uint8)
        truth = fr[:, 8:]
        tf_err += (pred != truth).sum(axis=0)
        roll = m.rollout(x[:, :8].unsqueeze(1)).cpu().numpy().astype(np.uint8)
        bad = roll != truth                                   # (B,8,G,G)
        roll_err += bad.sum(axis=0)
        exact_by_world.extend((~bad.reshape(len(fr), -1).any(axis=1)).tolist())
        exact += int((~bad.reshape(len(fr), -1).any(axis=1)).sum())
        for b in range(len(fr)):
            fb = np.flatnonzero(bad[b].reshape(8, -1).any(axis=1))
            if len(fb):
                k = int(fb[0])
                first_frame[k] += 1
                first_err += bad[b, k]
    s = seam_mask(G)
    tot_tf = int(tf_err.sum())
    tot_first = int(first_err.sum())
    return {
        "n": int(N), "rollout_seq_acc": exact / N,
        "tf_cell_acc": 1.0 - tot_tf / (N * 8 * G * G),
        "tf_errors": tot_tf,
        "tf_error_map": tf_err.sum(axis=0).tolist(),
        "tf_errors_by_frame": tf_err.reshape(8, -1).sum(axis=1).tolist(),
        "tf_seam_share": (int(tf_err.sum(axis=0)[s].sum()) / tot_tf) if tot_tf else None,
        "first_error_map": first_err.tolist(),
        "first_error_frame_hist": first_frame.tolist(),
        "first_error_cells": tot_first,
        "first_error_seam_share": (int(first_err[s].sum()) / tot_first) if tot_first else None,
        "seam_area_share": float(s.mean()),
        "tf_seam_error_rate": int(tf_err[:, s].sum()) / (N * 8 * int(s.sum())),
        "tf_interior_error_rate": int(tf_err[:, ~s].sum()) / (N * 8 * int((~s).sum())),
        "tf_error_rate_by_row": (tf_err.sum(axis=(0, 2)) / (N * 8 * G)).tolist(),
        "tf_error_rate_by_col": (tf_err.sum(axis=(0, 1)) / (N * 8 * G)).tolist(),
        "rollout_pixel_acc": 1.0 - int(roll_err.sum()) / (N * 8 * G * G),
        "rollout_error_map": roll_err.sum(axis=0).tolist(),
        "rollout_exact_by_world": exact_by_world,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_dir", default="data/eval_corpora")
    ap.add_argument("--experiment", default="E12.3")
    ap.add_argument("--output_name", default="e12_rope_seam.json")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    fp = code_fingerprint()
    script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    c = get_corpus(a.cache_dir, "l1_val_gol_g8", n=512, master_seed=43, half="train",
                   grid=8, rule_override=R.GOL_RULE)
    side = os.path.join(a.cache_dir, "l1_val_gol_g8.npz.json")
    sha = json.load(open(side)).get("sha256") if os.path.exists(side) else None
    for run in a.runs:
        t0 = time.time()
        torch.manual_seed(0)
        m, args, step = load(run, a.device)
        res = analyse(m, c["frames"], a.device, batch=a.batch)
        rec = {"run": run, "experiment": a.experiment, "ckpt_step": step, "weights": "ema",
               "pos_enc": args.get("pos_enc") or "learned", "code": fp,
               "eval_script": script_sha, "corpus": "l1_val_gol_g8",
               "corpus_sha256": sha, "eval_precision": "fp32", **res,
               "wall_s": time.time() - t0}
        d = os.path.join("data", "runs", run)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, a.output_name), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[{run}] pos {rec['pos_enc']:10s} rollout SeqAcc {res['rollout_seq_acc']:.4f} "
              f"tf-cell {res['tf_cell_acc']:.5f} tf-errors {res['tf_errors']} "
              f"seam share {res['tf_seam_share']} | first-error cells "
              f"{res['first_error_cells']} seam share {res['first_error_seam_share']} "
              f"(area {res['seam_area_share']:.4f})", flush=True)


if __name__ == "__main__":
    main()
