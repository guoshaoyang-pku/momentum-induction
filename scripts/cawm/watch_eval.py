"""Parallel ZS-rollout evaluator (protocol note 2026-08-29: training must
never block on eval — save a ckpt, evaluate it in a separate process).

Watches data/ckpt/<run>.pt; every time a new step appears it loads the
snapshot and runs the same rollout eval as train.py's inline eval, appending
"[eval watch N]" blocks (identical metric-line format) to
data/runs/<run>/watch_eval.log. When the final step is seen it also writes
data/runs/<run>/eval_final.json and exits. Checkpoint reads are safe because
train.py saves atomically (tmp + rename)."""
import argparse
import json
import os
import time

import torch

from cawm import rules as R
from cawm.train import (build_model, get_corpus, model_kwargs, primary_metric,
                        run_eval, save_best)


def build_corpora(cache_dir, task, grid, l4v2=None):
    """Mirror train.py's corpus construction exactly for each task."""
    if task == "L1":
        return {
            "l1_val": get_corpus(cache_dir, f"l1_val_gol_g{grid}", n=512,
                                 master_seed=43, half="train", grid=grid,
                                 rule_override=R.GOL_RULE),
        }
    if task == "L4" and l4v2 is not None:
        # E6.2/E6.3 constrained families (see train.py)
        return {
            "l4_zs": get_corpus(cache_dir, f"{l4v2}_zs_sc_g{grid}", n=2048,
                                master_seed=44, half="zs", grid=grid,
                                self_consistent=True, l4v2=l4v2),
            "l2_val": get_corpus(cache_dir, f"{l4v2}_val_sc_g{grid}", n=2048,
                                 master_seed=43, half="train", grid=grid,
                                 self_consistent=True, l4v2=l4v2),
        }
    if task == "L4":
        e, d, pr = 9, 1, 1.0
        return {
            "l4_zs": get_corpus(cache_dir, f"l4_zs_sc_e{e}_p{pr}_g{grid}",
                                n=2048, master_seed=44, half="zs", grid=grid,
                                l4=(e, d, pr), self_consistent=True, prior_slots=(e,)),
            "l4_zs_legacy": get_corpus(cache_dir, f"l4_zs_e{e}_p{pr}_g{grid}",
                                       n=2048, master_seed=44, half="zs", grid=grid,
                                       l4=(e, d, pr)),
            "l2_val": get_corpus(cache_dir, f"l4_val_e{e}_p{pr}_g{grid}",
                                 n=2048, master_seed=43, half="train", grid=grid,
                                 l4=(e, d, pr), self_consistent=True, prior_slots=(e,)),
        }
    # L23
    return {
        "l2_val": get_corpus(cache_dir, f"l2_val_g{grid}", n=2048,
                             master_seed=43, half="train", grid=grid),
        "l2_val_sc": get_corpus(cache_dir, f"l2_val_sc_g{grid}", n=2048,
                                master_seed=43, half="train", grid=grid,
                                self_consistent=True),
        # Legacy unfiltered corpus retained for historical comparison. The
        # older 0.8099 statistic is not an optimal SeqAcc ceiling; the SC
        # corpus is the direct-evidence primary evaluation.
        "l3_zs": get_corpus(cache_dir, f"l3_zs_g{grid}", n=2048,
                            master_seed=44, half="zs", grid=grid),
        "l3_zs_sc": get_corpus(cache_dir, f"l3_zs_sc_g{grid}", n=2048,
                               master_seed=44, half="zs", grid=grid,
                               self_consistent=True),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--task", choices=["L1", "L23", "L4"], default="L23")
    p.add_argument("--total_steps", type=int, required=True,
                   help="exit after the ckpt reaches this step")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--poll_s", type=float, default=30.0)
    p.add_argument("--cache_dir", default="data/eval_corpora")
    p.add_argument("--l4v2", default=None, choices=["l4a", "l4b"],
                   help="mirror train.py --l4v2 (task L4 constrained family)")
    p.add_argument("--device", default="cpu",
                   help="eval device; frame_ar-routed diffusion eval is ~8x "
                        "slower than uniform — use cuda to keep the node's "
                        "CPU load sane")
    p.add_argument("--bf16-eval", dest="bf16_eval", action="store_true",
                   default=True,
                   help="bf16 autocast for the rollout forward passes (DEFAULT "
                        "ON since 2026-09-19, matching train.py). Rollout is AR "
                        "(8 steps; 8x49 chain levels for diffusion), so this is "
                        "where eval wall time actually goes.")
    p.add_argument("--no-bf16-eval", dest="bf16_eval", action="store_false",
                   help="fp32 rollouts. Use when re-evaluating a run whose "
                        "sibling arms were harvested under fp32 eval, so the "
                        "comparison stays on one eval precision.")
    args = p.parse_args()

    ckpt_path = os.path.join("data", "ckpt", f"{args.run}.pt")
    out_dir = os.path.join("data", "runs", args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "watch_eval.log")

    corpora = build_corpora(args.cache_dir, args.task, args.grid, l4v2=args.l4v2)
    last_step, last_mtime, last_results = 0, 0.0, None
    best = {"metric": -1.0, "step": 0}
    while True:
        time.sleep(args.poll_s)
        if not os.path.exists(ckpt_path):
            continue
        mtime = os.path.getmtime(ckpt_path)
        if mtime <= last_mtime:
            continue
        ck = torch.load(ckpt_path, map_location="cpu")
        step = int(ck["step"])
        last_mtime = mtime
        if step <= last_step:
            continue
        cargs = ck["args"]
        # Architecture comes from the checkpoint's own recorded args via the
        # shared mapping. Never hand-copy build_model kwargs here: doing so is
        # how every E8.4/E7.2 arm became unscorable (the watcher rebuilt the
        # default ladder and failed to load the trained state_dict).
        model = build_model(
            cargs["seed"], model=cargs.get("model", "constructive"),
            **model_kwargs(cargs))
        # prefer EMA weights when present (recipe-v2 eval uses EMA)
        state = ck.get("model_ema", ck["model"])
        model.load_state_dict(state)
        model.to(args.device)
        with open(out_path, "a") as f:
            last_results = run_eval(model, corpora, args.device,
                                    f"watch {step}",
                                    l4v2=args.l4v2, bf16=args.bf16_eval,
                                    cov_bins=(args.task == "L23"),
                                    log=lambda m, f=f: (f.write(m + "\n"),
                                                        print(m, flush=True)))
        # best-ckpt retention (SETTINGS §9): with --eval_every 0 the training
        # process runs no inline eval, so the watcher (the process that sees
        # run_eval results) owns best-val saving for externally-watched runs.
        pm = primary_metric(last_results, args.task)
        if pm is not None and pm > best["metric"]:
            best["metric"], best["step"] = pm, step
            save_best(os.path.join("data", "ckpt", f"{args.run}_best.pt"),
                      model, step, last_results)
            print(f"[best] watch step {step}: primary {pm:.4f}", flush=True)
        last_step = step
        if step >= args.total_steps:
            break
    with open(os.path.join(out_dir, "eval_final.json"), "w") as f:
        json.dump({"run": args.run, "step": last_step, "final": last_results}, f)
    print(f"[watch] {args.run}: final step {last_step} evaluated, exiting")


if __name__ == "__main__":
    main()
