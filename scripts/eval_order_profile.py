"""E11 order profile (zero training; SETTINGS S13, EXPERIMENT_PLAN E11).

Evaluates the SAME weights of each named run under joint vs frame-ordered
sampling and records exactness frame by frame. No weight changes, no
checkpoint selection: the final-step checkpoint's EMA weights are used unless
--ckpt_suffix names another file (e.g. _best, for provenance cross-checks).

Samplers (per model kind):
  uniform          every canvas frame shares each chain level (joint denoising)
  frame_ar_clean   frame k runs its full chain, is hard-committed, then k+1
                   starts; not-yet-started frames are declared level 0 (the
                   registered E10 convention)
  frame_ar_noise   same order, not-yet-started frames declared the top level
                   (E10.5: the in-distribution input for a bidirectional
                   denoiser trained with per-frame levels)
  window{w}_noise  sliding window of w frames in flight (E10.4 definition)
  frame_ar / frame_ar_commit   structured DiffusionModel (E3.7): the D0 pair
Denoisers trained with one shared level (diag arms) only admit `uniform`.

Output: data/runs/<run>/order_profile<tag>.json (per sampler: SeqAcc, pixel
accuracy, frame-perfect rate and pixel accuracy for each of the 8 predicted
frames) and data/runs/<run>/order_examples<tag>.npz (the first --dump worlds:
prefix, truth and each sampler's prediction; real rollouts for figures).

Usage (repo root):
  PYTHONPATH=scripts python3 scripts/eval_order_profile.py --device cuda \
      --runs diffvan_L1_sc_s42_bf16 billiard_cubeD8_g16_s42 [--windows 2 4]
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from cawm import rules as R
from cawm.billiard import build_eval_set
from cawm.models import DiffusionVanilla
from cawm.train import build_model, code_fingerprint, get_corpus, model_kwargs


def load_run(run, suffix, device):
    final = torch.load(os.path.join("data", "ckpt", f"{run}.pt"),
                       map_location="cpu", weights_only=False)
    args = final["args"]
    if suffix:
        ck = torch.load(os.path.join("data", "ckpt", f"{run}{suffix}.pt"),
                        map_location="cpu", weights_only=False)
        sd, step = ck["model"], int(ck.get("step", -1))
    else:
        sd = {**final["model"], **final.get("model_ema", {})}
        step = int(final.get("step", -1))
    if args.get("world") == "billiard":
        m = DiffusionVanilla(grid=args["grid"], arm=args["arm"],
                             t_per_frame=args["billiard_arm"] == "cube",
                             n_layers=args["billiard_depth"])
        kind = "vanilla"
    else:
        m = build_model(args["seed"], model=args["model"], **model_kwargs(args))
        kind = "vanilla" if args["model"] == "diffusion_vanilla" else "structured"
    m.load_state_dict(sd)
    m = m.to(device).eval()
    return m, args, kind, step, "ema" if (not suffix and "model_ema" in final) else "stored"


def corpora_for(args, cache_dir, billiard_n):
    grid = args["grid"]
    if args.get("world") == "billiard":
        frames, sha = build_eval_set(billiard_n, grid=grid,
                                     n_balls=args.get("billiard_balls", 3),
                                     collide=bool(args.get("billiard_collide")))
        return {"billiard_eval": (frames, sha)}
    out = {}
    stride = int(args.get("frame_stride") or 1)
    if args["task"] == "L1" and stride == 1:
        c = get_corpus(cache_dir, f"l1_val_gol_g{grid}", n=512, master_seed=43,
                       half="train", grid=grid, rule_override=R.GOL_RULE)
        out[f"l1_val_gol_g{grid}"] = c
    elif args["task"] == "L1":
        # E11.5 strided corpora: n=2048 so the non-static stratum stays large
        name = f"l1_val_gol_s{stride}_n2048_g{grid}"
        out[name] = get_corpus(cache_dir, name, n=2048, master_seed=43,
                               half="train", grid=grid, rule_override=R.GOL_RULE,
                               stride=stride)
    elif args["task"] == "L23":
        c = get_corpus(cache_dir, f"l3_zs_sc_g{grid}", n=2048, master_seed=44,
                       half="zs", grid=grid, self_consistent=True)
        out[f"l3_zs_sc_g{grid}"] = c
    else:
        raise ValueError(f"task {args['task']} not in the E11 protocol")
    res = {}
    for name, c in out.items():
        sidecar = os.path.join(cache_dir, name + ".npz.json")
        sha = (json.load(open(sidecar)).get("sha256") if os.path.exists(sidecar)
               else hashlib.sha256(c["frames"].tobytes()).hexdigest())
        res[name] = (c["frames"], sha)
    return res


def budget_nfe(spec, budget):
    """Levels per chain so that the rollout spends at most `budget` denoiser
    calls (SETTINGS S13 amendment E11.4): joint = one chain; frame by frame
    = 8 sequential chains; window w = ceil(8/w) sequential groups."""
    if "window" in spec:
        groups = -(-8 // spec["window"])
    elif spec.get("schedule") == "frame_ar":
        groups = 8
    else:
        groups = 1
    return min(budget // groups, 49)


def samplers_for(model, kind, windows):
    if kind == "structured":
        return [("uniform", dict(schedule="uniform")),
                ("frame_ar", dict(schedule="frame_ar", commit=False)),
                ("frame_ar_commit", dict(schedule="frame_ar", commit=True))]
    if not model.t_per_frame:
        return [("uniform", dict(schedule="uniform"))]
    s = [("uniform", dict(schedule="uniform")),
         ("frame_ar_clean", dict(schedule="frame_ar", commit=True,
                                 pending="clean")),
         ("frame_ar_noise", dict(schedule="frame_ar", commit=True,
                                 pending="noise"))]
    for w in windows:
        s.append((f"window{w}_noise", dict(window=w, pending="noise")))
    return s


@torch.no_grad()
def roll(model, frames, device, spec, batch):
    preds = []
    for i in range(0, frames.shape[0], batch):
        x = torch.as_tensor(frames[i:i + batch, :8].astype(np.float32),
                            device=device)
        prefix = (2 * x - 1)[:, None]
        if "window" in spec:
            p = model.rollout_window(prefix, **spec)
        else:
            p = model.rollout(prefix, **spec)
        preds.append(p.cpu().numpy().astype(np.uint8))
    return np.concatenate(preds, axis=0)


def profile(preds, truth, last_hist=None):
    ok = preds == truth
    n = ok.shape[0]
    fp = ok.reshape(n, 8, -1).all(axis=2)
    exact = ok.reshape(n, -1).all(axis=1)
    extra = {}
    if last_hist is not None:
        # non-static stratum: worlds whose future is not the last history
        # frame repeated (dead boards, still lifes, oscillators whose period
        # divides the frame stride)
        static = (truth == last_hist[:, None]).reshape(n, -1).all(axis=1)
        extra = {"n_nonstatic": int((~static).sum()),
                 "seq_acc_nonstatic": (float(exact[~static].mean())
                                       if (~static).any() else None)}
    return {**extra, "seq_acc": float(exact.mean()),
            "pixel_acc": float(ok.mean()),
            "frame_perfect": [float(fp[:, k].mean()) for k in range(8)],
            "frame_pixel": [float(ok[:, k].mean()) for k in range(8)],
            "n": int(n)}


@torch.no_grad()
def evaluate_commit_modes(a):
    """E16 frozen one-call commitment controls; existing final EMA weights."""
    from cawm.commit_schedule import commit_rollout

    def sha_file(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    modes = ("joint", "causal", "confident", "random", "parallel")
    for run in a.runs:
        started = time.time()
        model, args, kind, step, weights = load_run(run, "", a.device)
        assert kind == "vanilla" and model.t_per_frame and model.n_layers == 8
        assert weights == "ema" and model.trainable_param_count() == 196417
        assert not args.get("billiard_collide", False)
        record = dict(experiment="E16", run=run, ckpt_step=step, weights=weights,
                      params=model.trainable_param_count(), depth=8,
                      code=code_fingerprint(), script_sha256=sha_file(__file__),
                      checkpoint_sha256=sha_file(f"data/ckpt/{run}.pt"),
                      eval_precision="fp32", sample_seed=a.sample_seed,
                      order_seed=0, batch=a.batch, threshold=.99, corpora={})
        arrays = {}
        for cname, (frames, sha) in corpora_for(args, a.cache_dir, a.billiard_n).items():
            truth = frames[:, 8:]
            crec = dict(sha256=sha, samplers={})
            for mode in modes:
                torch.manual_seed(a.sample_seed)
                if str(a.device).startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                batches, rounds, per_world_calls = [], [], []
                actual_calls = [0]
                tie_counts = [0, 0]
                original = model.denoise

                def counted(x, hist, levels):
                    actual_calls[0] += 1
                    u, v = original(x, hist, levels)
                    if mode in ("confident", "parallel"):
                        score = torch.tanh(u / 2).abs().mean(dim=(2, 3))
                        pending = levels > 0
                        score = score.masked_fill(~pending, float("-inf"))
                        tied = ((score == score.max(dim=1, keepdim=True).values)
                                & pending).sum(dim=1) > 1
                        tie_counts[0] += int(tied.sum())
                        tie_counts[1] += len(score)
                    return u, v

                model.denoise = counted
                try:
                    for i in range(0, len(frames), a.batch):
                        xb = torch.as_tensor(frames[i:i+a.batch, :8].astype(np.float32),
                                             device=a.device)
                        prefix = (2 * xb - 1)[:, None]
                        if mode == "joint":
                            p = model.rollout(prefix, nfe=8, schedule="uniform")
                            rr = torch.full((len(xb), 8), -1, dtype=torch.long,
                                            device=a.device)
                            cc = torch.full((len(xb),), 8, dtype=torch.long,
                                            device=a.device)
                        else:
                            p, rr, cc, nc = commit_rollout(model, prefix, mode,
                                                          threshold=.99, order_seed=i)
                            assert nc == int(cc.max())
                        batches.append(p.cpu().numpy())
                        rounds.append(rr.cpu().numpy())
                        per_world_calls.append(cc.cpu().numpy())
                finally:
                    model.denoise = original
                if str(a.device).startswith("cuda"):
                    torch.cuda.synchronize()
                pred = np.concatenate(batches)
                rr = np.concatenate(rounds)
                cc = np.concatenate(per_world_calls)
                exact = (pred == truth).reshape(len(truth), -1).all(axis=1)
                metrics = profile(pred, truth, frames[:, 7])
                metrics.update(mean_calls=float(cc.mean()), min_calls=int(cc.min()),
                               max_calls=int(cc.max()), actual_batch_forward_calls=actual_calls[0],
                               wall_s=time.time()-t0,
                               peak_gpu_memory_bytes=(torch.cuda.max_memory_allocated()
                                 if str(a.device).startswith("cuda") else None),
                               confidence_tie_world_rounds=tie_counts[0],
                               scored_world_rounds=tie_counts[1])
                if mode != "joint":
                    metrics["mean_frames_committed_per_round"] = [
                        float((rr == j).sum(axis=1).mean()) for j in range(8)]
                    metrics["exact_causal_order_share"] = float(
                        (rr == np.arange(8)[None]).all(axis=1).mean())
                crec["samplers"][mode] = metrics
                for field, array in (("predictions", pred), ("exact", exact),
                                     ("commit_round", rr), ("calls", cc)):
                    arrays[f"{cname}__{mode}__{field}"] = array
                print(f"[{run}] {mode} SeqAcc={metrics['seq_acc']:.6f} "
                      f"nonstatic={metrics.get('seq_acc_nonstatic')} "
                      f"calls={cc.mean():.3f} ties={tie_counts}", flush=True)
            record["corpora"][cname] = crec
        record["wall_s"] = time.time() - started
        out = os.path.join("data", "runs", run)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "e16_commit_order")
        if os.path.exists(path + ".json"):
            raise FileExistsError(path + ".json")
        np.savez_compressed(path + ".npz", **arrays)
        with open(path + ".json", "w") as f:
            json.dump(record, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt_suffix", default="",
                    help="'' = final-step ckpt (EMA); '_best' for provenance")
    ap.add_argument("--windows", nargs="*", type=int, default=[])
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--billiard_n", type=int, default=512)
    ap.add_argument("--dump", type=int, default=64)
    ap.add_argument("--budget", type=int, default=None,
                    help="E11.4: total denoiser calls per rollout; overrides "
                         "--nfe per sampler (joint 1 chain, frame by frame 8)")
    ap.add_argument("--nfe", type=int, default=49,
                    help="denoising steps per chain (49 = full chain, the "
                         "registered default; <49 = DDIM sub-sampled levels)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache_dir", default="data/eval_corpora")
    ap.add_argument("--adaptive_order", action="store_true",
                    help="E16: fixed one-call confidence/parallel commitment")
    a = ap.parse_args()
    if a.adaptive_order:
        assert a.ckpt_suffix == "" and a.sample_seed == 0 and a.batch == 256
        evaluate_commit_modes(a)
        return
    fp = code_fingerprint()
    script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    for run in a.runs:
        t0 = time.time()
        model, args, kind, step, weights = load_run(run, a.ckpt_suffix, a.device)
        rec = {"run": run, "kind": kind, "ckpt_step": step, "weights": weights,
               "ckpt_suffix": a.ckpt_suffix, "code": fp, "eval_script": script_sha,
               "eval_precision": "fp32",
               "sample_seed": a.sample_seed, "nfe": a.nfe, "budget": a.budget,
               "params_trainable": model.trainable_param_count(),
               "t_per_frame": bool(getattr(model, "t_per_frame", False)),
               "depth": getattr(model, "n_layers", None), "corpora": {}}
        examples = {}
        for cname, (frames, sha) in corpora_for(args, a.cache_dir,
                                                a.billiard_n).items():
            truth = frames[:, 8:]
            crec = {"sha256": sha, "samplers": {}}
            examples[f"{cname}__prefix"] = frames[:a.dump, :8]
            examples[f"{cname}__truth"] = truth[:a.dump]
            for sname, spec in samplers_for(model, kind, a.windows):
                nfe = a.nfe if a.budget is None else budget_nfe(spec, a.budget)
                if nfe < 1:
                    continue
                calls = [0]
                orig = model.denoise

                def counted(*args, _o=orig, **kw):
                    calls[0] += 1
                    return _o(*args, **kw)

                model.denoise = counted
                torch.manual_seed(a.sample_seed)
                preds = roll(model, frames, a.device, {**spec, "nfe": nfe},
                             a.batch)
                model.denoise = orig
                n_batches = -(-frames.shape[0] // a.batch)
                m = profile(preds, truth, frames[:, 7])
                m["nfe_per_chain"] = nfe
                m["calls_per_rollout"] = calls[0] // n_batches
                crec["samplers"][sname] = m
                examples[f"{cname}__{sname}"] = preds[:a.dump]
                # per-world exactness for every world (post-hoc stratification)
                examples[f"{cname}__{sname}__exact"] = (
                    preds == truth).reshape(len(truth), -1).all(axis=1)
                print(f"[{run}] {cname} {sname:16s} calls {m['calls_per_rollout']:3d} "
                      f"SeqAcc {m['seq_acc']:.4f} "
                      f"nonstatic {m['seq_acc_nonstatic'] if m['seq_acc_nonstatic'] is None else round(m['seq_acc_nonstatic'], 4)} "
                      f"Pix {m['pixel_acc']:.4f} framePerfect "
                      f"{[round(v, 3) for v in m['frame_perfect']]}", flush=True)
            rec["corpora"][cname] = crec
        rec["wall_s"] = time.time() - t0
        out_dir = os.path.join("data", "runs", run)
        os.makedirs(out_dir, exist_ok=True)
        tag = a.tag + a.ckpt_suffix
        with open(os.path.join(out_dir, f"order_profile{tag}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        np.savez_compressed(os.path.join(out_dir, f"order_examples{tag}.npz"),
                            **examples)
        print(f"[{run}] done {rec['wall_s']:.0f}s params={rec['params_trainable']} "
              f"step={step} weights={weights} code={fp}", flush=True)


if __name__ == "__main__":
    main()
