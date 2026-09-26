"""E10 billiard world: external validation of ordered deciding (2026-09-23).

Preregistered in docs/EXPERIMENT_PLAN.md (E10) and docs/SETTINGS.md S12 BEFORE
any code or launch. World: 3 point balls on a 16x16 torus, each with a constant
velocity drawn from the 8 nonzero king moves; binary OR-occupancy frames;
16 frames per trajectory (8 history + 8 future, the CA layout). Deterministic
and Markovian in the latent state; each ball's velocity is inferable from the
8-frame history (overlap events merge occupancies but not velocities, which
stay constant and trackable).

Arm U ("cube") trains DiffusionVanilla with per-frame independent noise levels
(the schedule cube) and evals the SAME EMA weights under two sampling paths:
the uniform diagonal (all frames share each chain level) and the frame-ordered
staircase with commit -- the CA sampler-swap replica. Arm D ("diag") trains
with the legacy shared scalar level (the standard video-diffusion recipe) as
the CA generic-denoiser analogue.
"""

import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from .models import DiffusionVanilla

MOVES = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
T_IN = 8
T_OUT = 8
TRAIN_MASTER_SEED = 42
EVAL_MASTER_SEED = 4242
EVAL_N = 512


def _world_params(index, master_seed, grid, n_balls):
    """Deterministic per-index draw; order = positions then velocities (S12)."""
    rng = np.random.default_rng([int(master_seed), int(index)])
    cells = rng.choice(grid * grid, size=n_balls, replace=False)
    pos = np.stack([cells // grid, cells % grid], axis=1).astype(np.int64)
    vel = np.array([MOVES[i] for i in rng.integers(0, len(MOVES), size=n_balls)],
                   dtype=np.int64)
    return pos, vel


def simulate_world(pos, vel, steps, grid, collide=False):
    """Roll `steps` transitions. Returns (steps+1, grid, grid) uint8 frames.

    collide (E10.4): annihilation -- balls sharing a cell after a move are ALL
    removed from the world. (Velocity permutation, the first candidate, is
    unobservable under OR-occupancy: identical-particle ghost equivalence.)
    Observable: a transient merged frame followed by disappearance; inferable:
    approaching trajectories are visible in the history."""
    frames = []
    p = np.asarray(pos, dtype=np.int64).copy()
    alive = np.ones(len(p), dtype=bool)
    for _ in range(steps + 1):
        f = np.zeros((grid, grid), dtype=np.uint8)
        if alive.any():
            f[p[alive, 0], p[alive, 1]] = 1          # OR-occupancy
        frames.append(f)
        if collide and alive.any():
            # annihilate AFTER recording: the merged frame stays visible for
            # one step, then all sharing balls vanish together
            keys = p[alive, 0] * grid + p[alive, 1]
            u, c = np.unique(keys, return_counts=True)
            if (c > 1).any():
                dead = {int(k) for k, n in zip(u, c) if n > 1}
                idx = np.where(alive)[0]
                alive[idx] &= np.array(
                    [int(p[i, 0] * grid + p[i, 1]) not in dead for i in idx])
        p = (p + vel) % grid
    return np.stack(frames, axis=0)


def collision_rate(frames, n_balls):
    """Fraction of trajectories with >=1 post-update shared cell (merge)."""
    return overlap_rate(frames, n_balls)


def sample_frames(indices, master_seed, grid=16, n_balls=3, collide=False):
    """Batch of trajectories (B, 16, grid, grid) uint8, index-addressed."""
    out = np.empty((len(indices), T_IN + T_OUT, grid, grid), dtype=np.uint8)
    for j, i in enumerate(indices):
        pos, vel = _world_params(i, master_seed, grid, n_balls)
        out[j] = simulate_world(pos, vel, T_IN + T_OUT - 1, grid,
                                collide=collide)
    return out


def overlap_rate(frames, n_balls):
    """Fraction of trajectories with >=1 frame of merged occupancy."""
    merged = (frames.reshape(frames.shape[0], frames.shape[1], -1).sum(-1)
              < n_balls).any(axis=1)
    return float(merged.mean())


def build_eval_set(n=EVAL_N, master_seed=EVAL_MASTER_SEED, grid=16, n_balls=3,
                   collide=False):
    frames = sample_frames(range(n), master_seed, grid, n_balls, collide)
    sha = hashlib.sha256(frames.tobytes()).hexdigest()
    return frames, sha


def rollout_metrics(preds, truth):
    """preds/truth: (B, 8, H, W) uint8 (future frames). Strict + soft metrics."""
    ok = (preds == truth)
    seq = ok.reshape(ok.shape[0], -1).all(axis=1).astype(np.float64)
    frame_perf = ok.reshape(ok.shape[0], ok.shape[1], -1).all(axis=2)
    return {
        "seq_acc": float(seq.mean()),
        "pixel_acc": float(ok.mean()),
        "frame_perfect": [float(frame_perf[:, k].mean()) for k in range(8)],
        "first_frame_pixel": float(ok[:, 0].mean()),
    }


@torch.no_grad()
def _rollout_sched(model, prefix, device, schedule, commit, batch=64):
    preds = []
    for i in range(0, prefix.shape[0], batch):
        x = torch.as_tensor(prefix[i:i + batch], device=device).float()
        x = 2 * x - 1
        p = model.rollout(x[:, None], schedule=schedule, commit=commit)
        preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def _rollout_window(model, prefix, device, window, polish=0, batch=64):
    preds = []
    for i in range(0, prefix.shape[0], batch):
        x = torch.as_tensor(prefix[i:i + batch], device=device).float()
        x = 2 * x - 1
        p = model.rollout_window(x[:, None], window=window, polish=polish)
        preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_billiard(args):
    """Self-contained E10 train + swap-eval entry (dispatched from train.py)."""
    from .train import EMA, code_fingerprint, lr_mult, save_ckpt

    assert args.billiard_arm in ("cube", "diag")
    cube = args.billiard_arm == "cube"
    run_name = args.out or f"billiard_{args.billiard_arm}_g{args.grid}_s{args.seed}"
    out_dir = os.path.join("data", "runs", run_name)
    ckpt_dir = os.path.join("data", "ckpt")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    device = args.device
    collide = bool(getattr(args, "billiard_collide", False))
    torch.manual_seed(args.seed)
    model = DiffusionVanilla(grid=args.grid, arm=args.arm,
                             t_per_frame=cube,
                             n_layers=args.billiard_depth).to(device)
    n_train, n_total = model.trainable_param_count(), model.total_param_count()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, args.ema) if args.ema > 0 else None

    fingerprint = code_fingerprint()
    log(f"[run] {run_name} -> {out_dir}")
    log(f"[code] cawm sha256:{fingerprint}")
    log(f"[world] billiard grid={args.grid} balls={args.billiard_balls} "
        f"torus, king-move velocities, OR-occupancy, {T_IN}+{T_OUT} frames "
        f"collide={collide}")
    log(f"[data] train stream master_seed={TRAIN_MASTER_SEED} (index-addressed); "
        f"eval stream master_seed={EVAL_MASTER_SEED} n={args.billiard_eval_n}")
    log(f"[recipe] arm={args.billiard_arm} b{args.batch} steps={args.steps} "
        f"lr={args.lr} clip={args.clip} cosine warmup={args.warmup} "
        f"ema={args.ema} pos_weight={args.billiard_pos_weight} "
        f"depth={args.billiard_depth} bf16_train={args.bf16} fp32_eval")
    log(f"[params] trainable={n_train} total={n_total}")

    eval_frames, eval_sha = build_eval_set(args.billiard_eval_n, grid=args.grid,
                                         n_balls=args.billiard_balls,
                                         collide=collide)
    log(f"[eval-set] sha256:{eval_sha} "
        f"merge_rate={overlap_rate(eval_frames, args.billiard_balls):.4f}")

    t0 = time.time()
    tail = []
    model.train()
    for step in range(1, args.steps + 1):
        idx = range((step - 1) * args.batch, step * args.batch)
        frames = sample_frames(idx, TRAIN_MASTER_SEED, args.grid,
                               args.billiard_balls, collide=collide)
        x = torch.as_tensor(frames, device=device).float()
        x = (2 * x - 1)[:, None]                          # (B,1,16,H,W)
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_mult(step, args.lr_schedule, args.warmup,
                                        args.steps)
        ctx = (torch.autocast("cuda", torch.bfloat16)
               if args.bf16 and device == "cuda" else
               torch.autocast("cpu", enabled=False))
        with ctx:
            loss, acc = model.training_loss(
                x, pos_weight=args.billiard_pos_weight)
        opt.zero_grad()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if ema is not None:
            ema.update(model)
        tail.append((float(loss.detach()), float(acc.detach())))
        tail = tail[-250:]
        if step % 250 == 0 or step == 1:
            el = time.time() - t0
            log(f"step {step}/{args.steps} loss {np.mean([t[0] for t in tail]):.4f} "
                f"tf_pixel {np.mean([t[1] for t in tail]):.4f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {el:.0f}s")

    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    save_ckpt(ckpt_path, model, opt, args.steps, {"tail": tail}, args, ema=ema)
    log(f"[ckpt] {ckpt_path}")

    truth = eval_frames[:, T_IN:]
    prefix = eval_frames[:, :T_IN]
    results = {}
    ctx = ema.swap(model) if ema is not None else _nullctx(model)
    with ctx:
        model.eval()
        schedules = [("uniform", False)]
        if cube:
            schedules.append(("frame_ar", True))
        for sched, commit in schedules:
            preds = _rollout_sched(model, prefix, device, sched, commit)
            m = rollout_metrics(preds, truth)
            results[sched] = m
            log(f"[eval final] {sched:9s} SeqAcc {m['seq_acc']:.4f} "
                f"Pix {m['pixel_acc']:.4f} ext1 {m['first_frame_pixel']:.4f} "
                f"framePerfect {[round(v, 3) for v in m['frame_perfect']]}")
        if cube:
            # E10.4 riding evals on the SAME EMA weights: window sweep,
            # frame_ar + polish, and the correction-capacity probe
            for w in (2, 4):
                preds = _rollout_window(model, prefix, device, window=w)
                m = rollout_metrics(preds, truth)
                results[f"window{w}"] = m
                log(f"[eval final] window{w:7d} SeqAcc {m['seq_acc']:.4f} "
                    f"Pix {m['pixel_acc']:.4f}")
            preds = _rollout_window(model, prefix, device, window=1, polish=8)
            m = rollout_metrics(preds, truth)
            results["frame_ar_polish8"] = m
            log(f"[eval final] far+polish SeqAcc {m['seq_acc']:.4f} "
                f"Pix {m['pixel_acc']:.4f}")
            # correction capacity: flip the lowest-index live cell of the
            # committed frame 3, polish, count restorations to truth
            committed = _rollout_window(model, prefix, device, window=1)
            fixed, total_rep, flips = 0, 0, []
            for i0 in range(0, prefix.shape[0], 64):
                sub = committed[i0:i0 + 64]
                for j in range(sub.shape[0]):
                    live = np.argwhere(sub[j, 3] == 1)
                    if len(live) == 0:
                        continue
                    y, xx = (int(v) for v in live[0])
                    flips.append((i0 + j, y, xx))
            repaired = 0
            for i0 in range(0, len(flips), 64):
                batch_idx = flips[i0:i0 + 64]
                rows = [b[0] for b in batch_idx]
                xp = torch.as_tensor(prefix[rows], device=device).float()
                xp = 2 * xp - 1
                # per-world inject: run one at a time (inject is scalar)
                outs = []
                for r, (gj, y, xx) in zip(range(xp.shape[0]), batch_idx):
                    o = model.rollout_window(
                        xp[r:r + 1, None], window=1, polish=8,
                        inject=(3, (y, xx)))
                    outs.append(o.cpu().numpy())
                    truth_cell = truth[gj, 3, y, xx]
                    repaired += int(outs[-1][0, 3, y, xx] == truth_cell)
            rep = repaired / max(len(flips), 1)
            results["correction"] = {"flips": len(flips), "repair_rate": rep}
            log(f"[eval final] correction repair {rep:.4f} over {len(flips)} flips")
    wall = time.time() - t0

    with open(os.path.join(out_dir, "eval_final.json"), "w") as f:
        json.dump({**results,
                   "eval_sha256": eval_sha,
                   "merge_rate": overlap_rate(eval_frames, args.billiard_balls),
                   "collide": collide,
                   "n": int(eval_frames.shape[0]), "grid": args.grid,
                   "balls": args.billiard_balls, "arm": args.billiard_arm,
                   "seed": args.seed, "steps": args.steps,
                   "pos_weight": args.billiard_pos_weight,
                   "eval_precision": "fp32", "code": fingerprint}, f, indent=2)
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump({"step": args.steps,
                   "loss_tail": float(np.mean([t[0] for t in tail])),
                   "tf_pixel_tail": float(np.mean([t[1] for t in tail])),
                   "wall_time_s": wall, "params_trainable": n_train,
                   "params_total": n_total, "code": fingerprint}, f, indent=2)
    log(f"[done] {run_name} wall {wall:.0f}s")


class _nullctx:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, *a):
        return False
