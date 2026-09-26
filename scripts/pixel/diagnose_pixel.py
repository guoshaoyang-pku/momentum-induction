"""Per-slot error anatomy of a trained pixel model (zero-training).

Loads a checkpoint, rolls out on the L23 eval set, de-renders predictions,
and buckets cell errors by the true slot of the queried transition.
"""
import argparse
import json

import numpy as np
import torch

from . import ca as ca_mod
from . import render as render_mod
from .models_pixel import build
from .stream import make_eval_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_traj", type=int, default=256)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location=a.device)
    cfg = ck["config"]
    model = build(cfg["model"], s=cfg["s"], frame=cfg["frame"],
                  width=cfg.get("width", 32), patch=cfg.get("patch", 16),
                  d=cfg.get("d", 64), attn_scale=cfg.get("attn_scale", 20.0),
                  key_bottleneck=cfg.get("key_bottleneck", False)).to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()

    batches = make_eval_batches(level="L23", s=cfg["s"], frame=cfg["frame"],
                                offsets=[(0, 0)], k_values=(1,),
                                n_traj=a.n_traj, p_init=cfg.get("p_init", 0.3))
    s = cfg["s"]
    slot_err = np.zeros(ca_mod.N_SLOTS)
    slot_cnt = np.zeros(ca_mod.N_SLOTS)
    traj_errs = []
    with torch.no_grad():
        for eb in batches:
            hist = eb["hist"].to(a.device)
            B = hist.shape[0]
            window = hist.clone()
            preds = []
            for _ in range(8):
                logits, _ = model(window)
                nxt = torch.where(logits >= 0, 1.0, -1.0)
                preds.append(nxt)
                window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], 1)
            preds = torch.stack(preds, 1).cpu().numpy()
            states = eb["states"].numpy()
            geom = eb["geom"].numpy()
            for b in range(B):
                oy, ox = geom[b]
                st_pred = np.stack([
                    render_mod.frame_to_cells(preds[b, j, 0], s, (oy, ox))
                    for j in range(8)])
                st_gt = states[b, 8:]
                errs = []
                for j in range(8):
                    slots = ca_mod.slots_of(states[b, 7 + j])  # query slots
                    bad = (st_pred[j] != st_gt[j])
                    for sl in range(ca_mod.N_SLOTS):
                        m = slots == sl
                        if m.any():
                            slot_cnt[sl] += m.sum()
                            slot_err[sl] += (bad & m).sum()
                    errs.append(bad.mean())
                traj_errs.append((float(np.mean(errs)),
                                  float((st_pred == st_gt).all())))
    order = np.argsort(-slot_err / np.maximum(slot_cnt, 1))
    print(json.dumps({"ckpt": a.ckpt, "n": a.n_traj,
                      "seqacc": float(np.mean([t[1] for t in traj_errs])),
                      "cell_err": float(np.mean([t[0] for t in traj_errs]))}))
    print("slot  own  n    err_rate   count")
    for sl in order:
        own, n = divmod(sl, 9)
        rate = slot_err[sl] / max(slot_cnt[sl], 1)
        print(f"{sl:3d}  {own:3d}  {n:2d}   {rate:8.5f}  {int(slot_cnt[sl])}")


if __name__ == "__main__":
    main()
