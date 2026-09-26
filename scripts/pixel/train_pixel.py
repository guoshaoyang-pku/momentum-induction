"""Pixel lab training loop. Standalone (does not import cawm), same recipe
family as the mainline RV2: AdamW + cosine to 10% with warmup, grad clip 1.0,
EMA 0.999 for eval and best-checkpoint selection.

Training: teacher-forced next-video-frame prediction over the future window.
Eval: autoregressive rollout in video space with hard sign feedback.
Metrics (grid region only, geometry known from generation):
  pix_acc    per-pixel sign agreement inside the grid region
  frame_ex   whole future frame (grid region) exactly right
  cell_acc   de-rendered cell accuracy (block average > 0)
  seqacc     all 8 future states x 64 cells exactly right (the mainline metric)
  traj_ex    all 8k future video frames exactly right
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import ca as ca_mod
from . import render as render_mod
from .models_pixel import build
from .stream import PixelStream, make_eval_batches

RULES = {"b2s3": ca_mod.RULE_B2S3, "gol": ca_mod.RULE_GOL,
         "highlife": ca_mod.RULE_HIGHLIFE}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="pcnn", choices=["pcnn", "part"])
    p.add_argument("--level", default="L1",
                   choices=["L1", "L23", "L4", "L4A", "L4B"])
    p.add_argument("--l1_rule", default="b2s3", choices=list(RULES))
    p.add_argument("--s", type=int, default=8)
    p.add_argument("--frame", type=int, default=128)
    p.add_argument("--offsets", default="center", choices=["center", "aligned"])
    p.add_argument("--k_max", type=int, default=1)
    p.add_argument("--border", type=int, default=1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--attn_scale", type=float, default=20.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--aux_lat", type=float, default=0.3)
    p.add_argument("--sup_perc", type=float, default=0.0,
                   help="part only: supervise patch embeddings with true cell "
                        "values and situation codes with true 18-slot labels")
    p.add_argument("--rule_pool", type=int, default=0,
                   help="L23: sample rules from a fixed pool of N (0 = i.i.d.)")
    p.add_argument("--init_from", default="",
                   help="warm-start model weights from a checkpoint 'model' key")
    p.add_argument("--lr_drop_at", type=int, default=0,
                   help="step at which lr drops to lr*lr_drop_factor (0=off)")
    p.add_argument("--lr_drop_factor", type=float, default=0.1)
    p.add_argument("--freeze_at", type=int, default=0,
                   help="part: freeze perception+retrieval at this step, "
                        "train only wv/head afterwards (0=off)")
    p.add_argument("--cell_loss_w", type=float, default=0.0,
                   help="aux BCE on cell-pooled logits (full-frame arms only); "
                        "restores mainline-like loss mass per cell")
    p.add_argument("--key_bottleneck", action="store_true",
                   help="part: keys/queries through a supervised 18-slot "
                        "softmax bottleneck (discrete mainline analog)")
    p.add_argument("--p_init_hi", type=float, default=0.0,
                   help="if > 0, sample init density per traj ~ U(p_init, hi)")
    p.add_argument("--ex_train", type=float, default=0.0,
                   help="L4 levels: fraction of each batch forced exercised "
                        "(mainline E6.6 dose intervention)")
    p.add_argument("--ex_pool", default="",
                   help="npy/npz of pre-built exercised trajectories "
                        "(states (M,16,8,8) uint8); used for ex_train slots")
    p.add_argument("--sup_values", action="store_true",
                   help="part: also supervise retrieved value tokens with the "
                        "true next-cell bit")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_traj", type=int, default=64)
    p.add_argument("--p_init", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="data/runs")
    return p.parse_args()


def lr_at(step, total, base, warmup, drop_at=0, drop_factor=0.1):
    if drop_at and step >= drop_at:
        return base * drop_factor
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, batches, s, hist_states, fut_states, device):
    model.eval()
    per_k = []
    for eb in batches:
        k = eb["k"]
        hist = eb["hist"].to(device)
        B = hist.shape[0]
        fut_len = fut_states * k
        window = hist.clone()
        preds = []
        for _ in range(fut_len):
            logits, _ = model(window)
            nxt = torch.where(logits >= 0, 1.0, -1.0)
            preds.append(nxt)
            window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], 1)
        preds = torch.stack(preds, 1)  # (B, fut_len, 1, F, F)
        fut = eb["fut"].to(device)
        states = eb["states"].cpu().numpy()
        geom = eb["geom"].numpy()
        pix_ok, fr_ok, cell_ok, seq_ok, tr_ok = [], [], [], [], []
        preds_np = preds.cpu().numpy()
        fut_np = fut.cpu().numpy()
        for b in range(B):
            oy, ox = geom[b]
            r = slice(oy, oy + 8 * s), slice(ox, ox + 8 * s)
            ok_pix = (preds_np[b, :, 0][(slice(None),) + r] ==
                      fut_np[b, :, 0][(slice(None),) + r]).mean()
            ok_fr = (preds_np[b, :, 0][(slice(None),) + r] ==
                     fut_np[b, :, 0][(slice(None),) + r]).all(axis=(1, 2)).mean()
            states_pred = np.stack([
                render_mod.frame_to_cells(preds_np[b, j * k + (k - 1), 0], s,
                                          (oy, ox))
                for j in range(fut_states)])
            states_gt = states[b, hist_states:]
            ok_cell = (states_pred == states_gt).mean()
            ok_seq = float((states_pred == states_gt).all())
            ok_tr = float((preds_np[b, :, 0][(slice(None),) + r] ==
                           fut_np[b, :, 0][(slice(None),) + r]).all())
            pix_ok.append(ok_pix); fr_ok.append(ok_fr); cell_ok.append(ok_cell)
            seq_ok.append(ok_seq); tr_ok.append(ok_tr)
        per_k.append({"k": k, "pix_acc": float(np.mean(pix_ok)),
                      "frame_ex": float(np.mean(fr_ok)),
                      "cell_acc": float(np.mean(cell_ok)),
                      "seqacc": float(np.mean(seq_ok)),
                      "traj_ex": float(np.mean(tr_ok))})
    model.train()
    return per_k


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.offsets == "center":
        c = (a.frame - 8 * a.s) // 2
        offsets = [(c, c)]
    else:
        offsets = render_mod.allowed_offsets(a.frame, 8, a.s)
    rule = RULES[a.l1_rule]
    ds = PixelStream(level=a.level, s=a.s, frame=a.frame, offsets=offsets,
                     k_max=a.k_max, batch=a.batch, seed=a.seed,
                     p_init=a.p_init, l1_rule=rule, border=a.border,
                     rule_pool=a.rule_pool, p_init_hi=a.p_init_hi,
                     ex_train=a.ex_train, ex_pool=a.ex_pool)
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers,
                    pin_memory=True)
    eval_batches = make_eval_batches(
        level=a.level, s=a.s, frame=a.frame, offsets=offsets,
        k_values=tuple(range(1, a.k_max + 1)), n_traj=a.eval_traj,
        p_init=a.p_init, l1_rule=rule, border=a.border)
    model = build(a.model, s=a.s, frame=a.frame, width=a.width,
                  patch=a.patch, d=a.d, attn_scale=a.attn_scale,
                  key_bottleneck=a.key_bottleneck).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    params = list(model.parameters())
    sit_probe = emb_probe = None
    if a.cell_loss_w > 0:
        assert a.offsets == "center" and a.frame == 8 * a.s, \
            "cell_loss needs full-frame geometry"
    if a.sup_perc > 0:
        assert a.model == "part" and a.frame // a.patch == 8, \
            "sup_perc needs part with patch grid == cell grid"
        if a.key_bottleneck:
            sit_probe = model.slot_head  # supervise the real bottleneck
        else:
            sit_probe = torch.nn.Linear(a.d, ca_mod.N_SLOTS).to(a.device)
            params += list(sit_probe.parameters())
        emb_probe = torch.nn.Linear(a.d, 1).to(a.device)
        params += list(emb_probe.parameters())
        val_probe = None
        if a.sup_values:
            val_probe = torch.nn.Linear(a.d, 1).to(a.device)
            params += list(val_probe.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    if a.init_from:
        ck = torch.load(a.init_from, map_location=a.device)
        model.load_state_dict(ck["model"])
        print(json.dumps({"init_from": a.init_from, "step": ck.get("step")}),
              flush=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log_path = os.path.join(a.out, f"{a.run}.jsonl")
    cfg = vars(a) | {"n_params": n_params, "offsets": len(offsets)}
    with open(log_path, "a") as f:
        f.write(json.dumps({"config": cfg}) + "\n")

    def eval_with_ema(step, loss_ema):
        raw_per_k = evaluate(model, eval_batches, a.s, 8, 8, a.device)
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        per_k = evaluate(model, eval_batches, a.s, 8, 8, a.device)
        model.load_state_dict(backup)
        row = {"step": step, "loss": round(loss_ema, 5),
               "eval": per_k, "eval_raw": raw_per_k,
               "t": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        score = np.mean([e["seqacc"] for e in per_k]) \
            + 0.01 * np.mean([e["frame_ex"] for e in per_k]) \
            + 1e-4 * np.mean([e["pix_acc"] for e in per_k])
        return score

    t0 = time.time()
    loss_ema, best = None, -1.0
    model.train()
    for step, batch in enumerate(dl):
        if step >= a.steps:
            break
        hist = batch["hist"].to(a.device, non_blocking=True)
        fut = batch["fut"].to(a.device, non_blocking=True)
        k = batch["k"]
        slotmaps = cellvals = None
        if a.sup_perc > 0:
            st_np = batch["states"].numpy()
            slotmaps = torch.from_numpy(np.stack([
                np.stack([ca_mod.slots_of(st_np[b, t]) for b in
                          range(st_np.shape[0])])
                for t in range(15)], 1)).long().to(a.device)
        if a.sup_perc > 0 or a.cell_loss_w > 0:
            st_np = batch["states"].numpy() if slotmaps is None else st_np
            cellvals = torch.from_numpy(st_np).float().to(a.device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup,
                            a.lr_drop_at, a.lr_drop_factor)
        if a.freeze_at and step == a.freeze_at:
            frozen = 0
            for n, prm in model.named_parameters():
                if not (n.startswith("head") or n.startswith("wv")):
                    prm.requires_grad_(False)
                    frozen += 1
            print(json.dumps({"froze": frozen, "step": step}), flush=True)
        window = hist
        losses = []
        for j in range(8 * k):
            if a.sup_perc > 0:
                logits, _, aux = model(window, return_aux=True)
            else:
                logits, lat = model(window)
            tgt = fut[:, j]
            l = F.binary_cross_entropy_with_logits(
                logits, (tgt + 1) / 2)
            if not a.sup_perc > 0 and lat is not None and a.aux_lat > 0:
                lat_tgt = F.avg_pool2d((tgt + 1) / 2, 2 ** model.levels)
                l = l + a.aux_lat * F.binary_cross_entropy_with_logits(
                    lat, lat_tgt)
            if a.cell_loss_w > 0:
                cl = F.avg_pool2d(logits, a.s).squeeze(1)   # (B,8,8)
                ct = cellvals[:, 8 + j // k]
                l = l + a.cell_loss_w * \
                    F.binary_cross_entropy_with_logits(cl, ct)
            if a.sup_perc > 0:
                T = window.shape[1]
                sidx = torch.tensor([(j + i) // k for i in range(T)],
                                    device=a.device)
                cell_lab = cellvals[:, sidx]                 # (B,T,8,8)
                e = aux["e"].permute(0, 1, 3, 4, 2)          # (B,T,Ph,Ph,d)
                l = l + a.sup_perc * F.binary_cross_entropy_with_logits(
                    emb_probe(e).squeeze(-1), cell_lab)
                m = sidx <= 14
                if m.any():
                    slot_lab = slotmaps[:, sidx[m]]          # (B,T',8,8)
                    s_ = aux["sit"][:, m].permute(0, 1, 3, 4, 2)
                    l = l + a.sup_perc * F.cross_entropy(
                        sit_probe(s_).reshape(-1, ca_mod.N_SLOTS),
                        slot_lab.reshape(-1))
                if a.sup_values:
                    v = aux["v"]                             # (B,T-1,Ph,Ph,d)
                    v_lab = cellvals[:, sidx[1:]]            # (B,T-1,8,8)
                    l = l + a.sup_perc * \
                        F.binary_cross_entropy_with_logits(
                            val_probe(v).squeeze(-1), v_lab)
            losses.append(l)
            window = torch.cat([window[:, 1:], tgt.unsqueeze(1)], 1)
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        with torch.no_grad():
            for key, v in model.state_dict().items():
                ema[key].mul_(a.ema).add_(v, alpha=1 - a.ema)
        lv = loss.item()
        loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            score = eval_with_ema(step + 1, loss_ema)
            if score > best:
                best = score
                torch.save({"model": ema,
                            "raw": {k: v.detach().clone()
                                    for k, v in model.state_dict().items()},
                            "config": cfg, "step": step + 1},
                           os.path.join(a.out, f"{a.run}_best.pt"))
    torch.save({"model": ema,
                "raw": {k: v.detach().clone()
                        for k, v in model.state_dict().items()},
                "config": cfg, "step": a.steps},
               os.path.join(a.out, f"{a.run}_last.pt"))
    print(json.dumps({"done": a.run, "best": best}), flush=True)


if __name__ == "__main__":
    main()
