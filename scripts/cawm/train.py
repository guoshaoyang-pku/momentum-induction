"""Single training entry point.

Tasks (docs/SETTINGS.md):
  L1  single-rule execution (default rule: Game of Life; --l1_rule to override)
  L23 thin multi-rule training on the train half; L2 = val metrics on seen
      rules, L3 = zero-shot metrics on the ZS half (one run, two reports)
  L4  default arbitration: rule distribution biased so entry --l4_entry takes
      value --l4_default with prob --l4_prior (default: entry 9 = (s=1,n=0),
      default 1, prior 1.0)

Usage: python -m cawm.train --task L23 --arm existence --steps 4000
"""

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from . import rules as R
from .evidence_loss import evidence_groups, evidence_bce
from .data import (StreamDataset, ChunkedStreamDataset, build_eval_corpus,
                   load_eval_corpus, unpredictable_cells_mask, PREFIX_LEN)
from .eval import (bf16_ctx, metrics, metrics_stratified_slot,
                   metrics_stratified_v2, metrics_cov_bins, rollout_corpus)
from .models.art_temporal import ArtTemporal
from .models import (ArtInduction, ArtKVShift, ArtTwoHop, ArtVanilla, ARCNN,
                     LifeGPTConfig, LifeGPT, AutomataGPTConfig, BurtsevConfigWM,
                     NCAConfigWM, DiffNCACfg,
                     ConstructiveCNN, DiffusionModel, DiffusionVanilla)

L4_ENTRY_DEFAULT = 9  # (s=1, n=0): isolated live cell


def code_fingerprint():
    """sha256 over all cawm package sources, logged per run: multi-node policy
    (author 2026-08-29) moves only code between clusters — this fingerprint is
    the verifiable 'code matched' canary for any result on any node."""
    h = hashlib.sha256()
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(pkg_dir, "**", "*.py"), recursive=True)):
        h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


def build_model(seed, model="constructive", **kw):
    """Construct the requested model with model init consuming a dedicated
    re-seeded generator, restoring the global RNG state afterwards.

    Author decision 2026-08-29: model seed and data seed are consumed
    separately (both 42 by default). Data streams are index-addressed numpy
    RNG and never touch torch RNG; this isolation makes model init invariant
    to DataLoader worker spawning (which draws its base seed from the global
    torch stream), independent of construction/iteration order. Bit-compatible
    with all earlier runs (init was already the first global-RNG consumer)."""
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    if model == "constructive":
        m = ConstructiveCNN(grid=kw["grid"], arm=kw["arm"], head=kw["head"],
                            det_feat=kw.get("det_feat", 3),
                            det_split=kw.get("det_split", 27),
                            agg_init=kw.get("agg_init", "ones"))
    elif model == "art":
        m = ArtInduction(grid=kw["grid"], arm=kw["arm"],
                         global_path=kw.get("art_global", False),
                         null_token=kw.get("art_null_token", False),
                         head_hidden=kw.get("art_head_hidden", 24),
                         global_norm=kw.get("art_global_norm", "none"),
                         reads=kw.get("art_reads", 1))
    elif model == "art_kvshift":
        m = ArtKVShift(grid=kw["grid"], arm=kw["arm"],
                       reads=kw.get("kv_reads", 1),
                       head_hidden=kw.get("kv_head_hidden", 24))
    elif model == "diffusion":
        m = DiffusionModel(grid=kw["grid"], arm=kw["arm"],
                           cascade_feedback=kw.get("cascade_feedback", "tanh"),
                           lookup_head=kw.get("lookup_head", "concat"),
                           corr_rf=kw.get("corr_rf", "fwd"),
                           t_per_frame=kw.get("t_per_frame", False),
                           no_cascade=kw.get("no_cascade", False),
                           corr_head=kw.get("corr_head", "mlp"),
                           prepend_hist=kw.get("prepend_hist", False),
                           spin_scale=kw.get("spin_scale", 1.0),
                           canvas_tanh=kw.get("canvas_tanh", False),
                           chain_nfe=kw.get("chain_nfe", 0),
                           chain_init=kw.get("chain_init", "noise"),
                           frontier=kw.get("frontier", False),
                           det_feat=kw.get("det_feat", 3),
                           det_split=kw.get("det_split", 27),
                           agg_init=kw.get("agg_init", "ones"),
                           cond_pairing=kw.get("cond_pairing", "conv"))
    elif model == "art_vanilla":
        m = ArtVanilla(grid=kw["grid"], arm=kw["arm"], head=kw["head"],
                       dmodel=kw.get("dmodel", 64),
                       nlayers=kw.get("nlayers"), nhead=kw.get("nhead"),
                       pos_enc=kw.get("pos_enc", "learned"),
                       tokenizer=kw.get("tokenizer", "cell"))
    elif model == "art_temporal":
        assert kw["arm"] == "emergence"
        m = ArtTemporal(grid=kw["grid"], head=kw["head"],
                        temporal=kw.get("temporal_support", "center"),
                        temp_mode=kw.get("twohop_temp", "learned"),
                        d=kw.get("twohop_d", 64), pos=kw.get("twohop_pos", "learned"))
    elif model == "art_twohop":
        m = ArtTwoHop(grid=kw["grid"], arm=kw["arm"], head=kw["head"],
                      temp_mode=kw.get("twohop_temp", "learned"),
                      d=kw.get("twohop_d", 64),
                      pos=kw.get("twohop_pos", "learned"))
    elif model == "arcnn":
        m = ARCNN(grid=kw["grid"], arm=kw["arm"], head=kw["head"])
    elif model == "lifegpt":
        m = LifeGPT(grid=kw["grid"], dmodel=kw.get("dmodel", 256),
                    nlayers=kw.get("nlayers") or 12,
                    nhead=kw.get("nhead") or 8,
                    head_dim=kw.get("life_head_dim", 64),
                    fcm=kw.get("life_fcm", 0.15))
    elif model == "lifegpt_cfg":
        m = LifeGPTConfig(grid=kw["grid"], arm=kw["arm"], head=kw["head"])
    elif model == "automatagpt_cfg":
        m = AutomataGPTConfig(grid=kw["grid"], arm=kw["arm"], head=kw["head"])
    elif model == "burtsev_cfg":
        m = BurtsevConfigWM(grid=kw["grid"], arm=kw["arm"], head=kw["head"])
    elif model == "nca_cfg":
        m = NCAConfigWM(grid=kw["grid"])
    elif model == "diffusion_vanilla":
        m = DiffusionVanilla(grid=kw["grid"], arm=kw["arm"], head=kw["head"],
                             t_per_frame=kw.get("t_per_frame", False),
                             n_layers=kw.get("vanilla_depth", 4))
    elif model == "diffnca_cfg":
        m = DiffNCACfg(grid=kw["grid"], arm=kw["arm"], head=kw["head"])
    else:
        raise ValueError(f"unknown model {model}")
    if kw.get("evidence_adapter", False):
        if model not in ("art", "art_kvshift"):
            raise ValueError("evidence_adapter only supported for ART/KV")
        m.evidence_adapter = torch.nn.Sequential(torch.nn.Linear(55,128), torch.nn.ReLU(),
            torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(m.evidence_adapter[-1].weight)
        torch.nn.init.zeros_(m.evidence_adapter[-1].bias)
    torch.set_rng_state(state)
    return m


def model_kwargs(cargs):
    """Map a training-args mapping (argparse namespace -> dict) to build_model
    kwargs. Shared by train.py and watch_eval.py.

    Why this exists (2026-09-20): the two used to hand-copy this list
    independently. E8.4 added `nlayers`/`nhead` and E7.2 added `pos_enc` to
    train.py only, so the watcher rebuilt the *default* ladder and could not load
    the checkpoint at all ("Missing key(s): blocks.2...blocks.5"). Every E8.4 and
    E7.2 arm trained fine and was permanently unscorable. A hand-copied list is a
    silent-divergence generator; one list is a single source of truth.

    Defaults here mirror build_model's own internal defaults, so a key that is
    absent from an older checkpoint's recorded args (flags that did not exist
    when that run was launched) resolves to the same value build_model would use.
    """
    g = cargs.get
    return dict(
        grid=cargs["grid"], arm=cargs["arm"], head=cargs["head"],
        cascade_feedback=g("cascade_feedback", "tanh"),
        lookup_head=g("lookup_head", "concat"),
        corr_rf=g("corr_rf", "fwd"),
        cond_pairing=g("cond_pairing", "conv"),
        t_per_frame=g("t_per_frame", False),
        vanilla_depth=g("vanilla_depth") or 4,
        no_cascade=g("no_cascade", False),
        corr_head=g("corr_head", "mlp"),
        prepend_hist=g("prepend_hist", False),
        spin_scale=g("spin_scale", 1.0),
        canvas_tanh=g("canvas_tanh", False),
        chain_nfe=g("chain_nfe", 0),
        chain_init=g("chain_init", "noise"),
        frontier=g("frontier", False),
        dmodel=g("dmodel", 64),
        temporal_support=g("temporal_support", "center"),
        twohop_temp=g("twohop_temp", "learned"),
        twohop_d=g("twohop_d", 64),
        twohop_pos=g("twohop_pos") or "learned",
        nlayers=g("nlayers"), nhead=g("nhead"),
        # None means "the flag did not exist when this run was launched"; the
        # registered default is the learned absolute table, and ArtVanilla
        # asserts on None, so coerce rather than pass it through.
        pos_enc=g("pos_enc") or "learned",
        tokenizer=g("tokenizer") or "cell",
        life_head_dim=g("life_head_dim", 64), life_fcm=g("life_fcm", 0.15),
        evidence_adapter=g("evidence_adapter", False),
        art_global=g("art_global", False),
        kv_reads=g("kv_reads", 1), kv_head_hidden=g("kv_head_hidden", 24),
        art_reads=g("art_reads", 1),
        art_null_token=g("art_null_token", False),
        art_head_hidden=g("art_head_hidden", 24),
        art_global_norm=g("art_global_norm", "none"),
        det_feat=g("det_feat", 3), det_split=g("det_split", 27),
        agg_init=g("agg_init", "ones"),
    )


def get_corpus(cache_dir, name, **kw):
    path = os.path.join(cache_dir, name + ".npz")
    if not os.path.exists(path):
        # Build to a PID-unique temp path, then publish with atomic renames.
        # Writing straight to `path` (the previous behaviour) let a second
        # process starting at the same time read a half-written archive and
        # die with EOFError; hit on 2026-09-19 when two E9.B arms launched
        # together on a node whose L4 corpora had not been pre-placed. The
        # published bytes are unchanged, so corpus sha256s are unaffected.
        # The temp name must keep the `.npz` suffix: np.savez_compressed
        # appends it when it is missing, which would strand the archive.
        tmp = f"{path}.tmp{os.getpid()}.npz"
        meta = build_eval_corpus(tmp, **kw)
        os.replace(tmp, path)
        os.replace(tmp + ".json", path + ".json")
        print(f"[corpus] built {name}: {meta['n']} trajs, draws={meta['draws']}, "
              f"sha256={meta['sha256'][:12]}")
    return load_eval_corpus(path)


def lr_mult(step, schedule, warmup, steps):
    """Learning-rate multiplier for `step` (1-based). const -> 1; cosine ->
    linear warmup 0->1 over `warmup` steps then cosine decay 1 -> 0.1 at
    `steps`."""
    if schedule == "const":
        return 1.0
    if warmup > 0 and step <= warmup:
        return step / warmup
    # cosine decay from 1 to 0.1 over the remaining steps
    t = (step - warmup) / max(1, steps - warmup)
    t = min(1.0, t)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))


class EMA:
    """Exponential-moving-average shadow of all trainable params (decay d).
    Update after every opt.step(): ema = d*ema + (1-d)*param. Eval and
    best-ckpt selection run under `swap()` (EMA weights swapped in, then raw
    weights restored); the final ckpt stores both raw and EMA."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    @contextlib.contextmanager
    def swap(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow})
        try:
            yield
        finally:
            model.load_state_dict({**model.state_dict(), **backup})


# Task -> ordered corpus-name candidates for the best-ckpt SELECTION metric.
# Selection must use TRAIN-HALF validation corpora only (l1_val; l2_val_sc /
# l2_val = SC-filtered train-half rules), never the zero-shot test corpora
# (l3_zs*, l4_zs*). Until 2026-09-02 this table pointed at l3_zs_sc / l4_zs_sc
# — test-set model selection; no reported headline number used a best-ckpt
# (all are final-step or last-watch numbers), but the _best.pt files written
# before that date were test-selected and must not be used for reporting.
PRIMARY_METRIC = {"L1": (["l1_val"], "seq_acc"),
                  "L23": (["l2_val_sc", "l2_val"], "seq_acc"),
                  "L4": (["l2_val"], "seq_acc")}


def primary_metric(results, task):
    """The task's best-ckpt selection SeqAcc (train-half validation) from a
    run_eval results dict, or None if no validation corpus was evaluated
    (no fallback to other corpora — a fallback could silently select on
    the test set)."""
    names, field = PRIMARY_METRIC[task]
    for name in names:
        if name in results and field in results[name]:
            return float(results[name][field])
    return None


def run_eval(model, corpora, device, tag, l4_entry=None, cov_bins=False,
             log=print, l4v2=None, bf16=False):
    lines = [f"[eval {tag}]"]
    results = {}
    is_diffusion = hasattr(model, "denoise")
    for name, corpus in corpora.items():
        preds = rollout_corpus(model, corpus, device=device, bf16=bf16)
        m = metrics(preds, corpus)
        if l4_entry is not None:
            m.update(metrics_stratified_slot(preds, corpus, l4_entry))
        if l4v2 is not None:
            m.update(metrics_stratified_v2(preds, corpus, l4v2))
        if cov_bins and name.startswith("l3"):
            m["cov_bins"] = metrics_cov_bins(preds, corpus)
        if is_diffusion:
            m_nc = metrics(rollout_corpus(model, corpus, device=device,
                                          bf16=bf16, ablate_canvas=True), corpus)
            m["seq_acc_no_canvas"] = m_nc["seq_acc"]
        results[name] = m
        parts = [f"SeqAcc {m['seq_acc']:.4f}", f"Pix {m['pixel_acc']:.4f}"]
        if "seq_acc_full_cov" in m:
            parts.append(f"fullCov {m['seq_acc_full_cov']:.4f}")
        if "seq_acc_slot_uncovered" in m:
            parts.append(f"slotUncov {m['seq_acc_slot_uncovered']:.4f}")
        if "seq_acc_default_exercised" in m:
            parts.append(f"defEx {m['seq_acc_default_exercised']:.4f}")
        if "seq_acc_derived_exercised" in m:
            parts.append(f"derEx {m['seq_acc_derived_exercised']:.4f}")
        if "seq_acc_no_canvas" in m:
            parts.append(f"noCanvas {m['seq_acc_no_canvas']:.4f}")
        lines.append(f"  {name:14s} " + "  ".join(parts))
    log("\n".join(lines))
    return results


def save_ckpt(path, model, opt, step, hist, args, ema=None):
    """Checkpoint with optimizer state so any run can be continued exactly
    (--resume). Same file is overwritten atomically (tmp + rename) so a
    parallel watch_eval reader never sees a torn file; legacy model-only
    ckpts resume as warm starts. When EMA is active the ckpt stores BOTH
    ck["model"] (raw) and ck["model_ema"]."""
    tmp = path + ".tmp"
    payload = {"model": model.state_dict(), "opt": opt.state_dict(),
               "step": step, "hist": hist, "args": vars(args)}
    if ema is not None:
        payload["model_ema"] = {k: v.detach().clone()
                                for k, v in ema.shadow.items()}
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_best(path, model, step, metric):
    """Atomically save the best-val checkpoint: the weights that were
    evaluated (EMA if active — the caller swaps them in before calling),
    the step, and the full metric dict. Best-ckpt retention (SETTINGS §9)."""
    tmp = path + ".tmp"
    torch.save({"model": {k: v.detach().clone()
                          for k, v in model.state_dict().items()},
                "step": step, "metric": metric}, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["L1", "L23", "L4"], required=True)
    p.add_argument("--model", choices=["constructive", "art", "art_kvshift",
                                       "diffusion", "art_vanilla", "art_twohop", "art_temporal",
                                       "lifegpt", "lifegpt_cfg", "automatagpt_cfg", "burtsev_cfg",
                                       "nca_cfg",
                                       "arcnn", "diffusion_vanilla",
                                       "diffnca_cfg"],
                   default="constructive")
    p.add_argument("--arm", choices=["existence", "emergence"], default="existence")
    p.add_argument("--head", choices=["concat", "bilinear"], default="concat",
                   help="constructive model only (ignored by art/diffusion)")
    # ---- diffusion stage-2 head/architecture flags (default = legacy) ----
    p.add_argument("--cascade_feedback", choices=["tanh", "sign"], default="tanh",
                   help="diffusion cascade feedback nonlinearity (legacy tanh)")
    p.add_argument("--lookup_head", choices=["concat", "bilinear"], default="concat",
                   help="diffusion lookup head (legacy concat)")
    p.add_argument("--cond_pairing", choices=["conv", "kvshift"], default="conv")
    p.add_argument("--corr_rf", choices=["fwd", "sym", "bwd"], default="fwd",
                   help="diffusion canvas-encoder temporal receptive field")
    p.add_argument("--t_per_frame", action="store_true",
                   help="diffusion: per-frame noise levels (default off)")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="E11.5 (SETTINGS S13): L1 only; show every s-th "
                        "generation, so each predicted frame is s rule "
                        "applications after the previous one. Default 1 = "
                        "registered protocol, bitwise.")
    p.add_argument("--vanilla_depth", type=int, default=4,
                   help="E11.2 (SETTINGS S13): diffusion_vanilla conv layers on "
                        "the CA world (temporal receptive field +-depth). "
                        "Default 4 = the registered generic denoiser, "
                        "state_dict unchanged.")
    p.add_argument("--no_cascade", action="store_true",
                   help="diffusion: skip the lookup cascade (default off)")
    # ---- diffusion E3.7 true-diffusion flags (default = legacy) ----
    p.add_argument("--corr_head", choices=["mlp", "bilinear"], default="mlp",
                   help="diffusion canvas-condition junction (legacy mlp)")
    p.add_argument("--prepend_hist", action="store_true",
                   help="diffusion: prepend clean hist[-1] as canvas frame 0")
    p.add_argument("--spin_scale", type=float, default=1.0,
                   help="diffusion: target spin magnitude (legacy 1 = ±1)")
    p.add_argument("--canvas_tanh", action="store_true",
                   help="diffusion: encoder reads tanh(canvas) belief input")
    p.add_argument("--t_frame_slope", type=float, default=0.0,
                   help="diffusion curriculum: per-frame noise offset slope*k")
    p.add_argument("--t_frame_slope_anneal", type=int, default=0,
                   help="steps to anneal t_frame_slope to 0 (0 = constant)")
    # ---- diffusion E3.8 chain-training flags (default = legacy/OFF) ----
    p.add_argument("--chain_nfe", type=int, default=0,
                   help="diffusion chain training: sub-sampled reverse-chain "
                        "levels for BOTH train and eval (0 = off = legacy "
                        "single-step marginal + full 49-level eval chain)")
    p.add_argument("--chain_init", choices=["zeros", "noise"], default="zeros",
                   help="chain start canvas: zeros (neutral belief, no info) "
                        "or noise (N(0,1) legacy start)")
    p.add_argument("--chain_aux", type=float, default=0.2,
                   help="chain training: per-step aux BCE weight at "
                        "intermediate levels (final level weight is 1.0)")
    p.add_argument("--frontier", action="store_true",
                   help="diffusion E3.9 frontier training: teacher-forced "
                        "clean prefix + noised frontier frame, BCE on the "
                        "frontier frame only; eval auto-routes to the "
                        "matched frame_ar+commit schedule (default off = "
                        "legacy)")
    p.add_argument("--art_global", action="store_true",
                   help="E6.5: parallel 36-d global evidence-count pathway into the ART head")
    p.add_argument("--kv_reads", type=int, choices=range(1, 9), default=1,
                   help="E17/E21: sequential KV-shift retrieval reads")
    p.add_argument("--art_reads", type=int, choices=range(1, 9), default=1,
                   help="E21: sequential retrieval reads of the pairing-conv ART")
    p.add_argument("--kv_head_hidden", type=int, default=24,
                   help="E17: KV-shift head width;87 matches two-read parameter count")
    p.add_argument("--art_null_token", action="store_true",
                   help="E6.5: learned attention-sink token (explicit no-evidence signal)")
    p.add_argument("--art_head_hidden", type=int, default=24,
                   help="E6.5: ART head MLP width (registered default 24)")
    p.add_argument("--art_global_norm", choices=["none", "log1p", "mean"], default="none",
                   help="E6.5: scaling of the ART global count vector")
    p.add_argument("--det_feat", type=int, default=3,
                   help="E6.6g: emergence detector featurizer channels (registered 3)")
    p.add_argument("--det_split", type=int, default=27,
                   help="E6.6g: emergence detector split width (registered 27)")
    p.add_argument("--agg_init", choices=["ones", "random", "positive"], default="ones",
                   help="E4.6: init of the emergence aggregation (pyramid + temporal "
                        "merge); 'ones' = registered analytic counting init (bitwise)")
    p.add_argument("--life_head_dim", type=int, default=64)
    p.add_argument("--life_fcm", type=float, default=0.15)
    p.add_argument("--microbatch", type=int, default=0,
                   help="LifeGPT only: split effective batch without changing data budget")
    p.add_argument("--dmodel", type=int, default=64, choices=[64, 128, 192, 256],
                   help="art_vanilla width (layers 2/4/6/8 respectively)")
    p.add_argument("--nlayers", type=int, default=None,
                   help="E8.4: art_vanilla depth override. Default None = the "
                        "registered width->depth mapping (bit-identical); set it "
                        "to break the depth/width tie the E4.1 ladder has")
    p.add_argument("--nhead", type=int, default=None,
                   help="E8.4: art_vanilla head-count override. Default None = "
                        "the registered 4 heads")
    p.add_argument("--pos_enc", choices=["learned", "factorized", "rope", "none",
                                            "rope_axial", "rope_torus",
                                            "rope_torus_mismatch", "rope_torus_detuned"],
                   default="learned",
                   help="E7.2: art_vanilla position-encoding control. 'learned' "
                        "(default) = the registered learned absolute table, "
                        "bit-identical; 'factorized' = separate frame/row/col "
                        "tables; 'rope' = rotary q/k, no additive table; "
                        "'none' (E12.6) = NoPE, the causal mask is the only "
                        "order signal")
    p.add_argument("--tokenizer", choices=["cell", "pair", "pair_shift"], default="cell",
                   help="A6: art_vanilla token embedding. 'cell' (default, "
                        "bit-identical) = the registered value embedding; "
                        "'pair' = adds a linear map of [3x3 neighbourhood of "
                        "the same cell in the previous frame, own value], "
                        "i.e. the situation-outcome pair built into the token")
    p.add_argument("--temporal_support", choices=["center", "forward", "backward", "symmetric"],
                   default="center", help="E18 art_temporal observed-prefix temporal support")
    p.add_argument("--twohop_temp", choices=["learned", "frozen20"], default="learned",
                   help="art_twohop attention temperature (E4.5 dose point)")
    p.add_argument("--twohop_d", type=int, default=64, choices=[64, 128],
                   help="art_twohop token width (E4.5 capacity dose point)")
    p.add_argument("--twohop_pos", choices=["learned", "none"], default="learned",
                   help="E12.2: art_twohop position tables. 'learned' (default) = "
                        "registered, bit-identical; 'none' = no positional signal")
    # ---- recipe-v2 flags (all default = legacy/OFF) ----
    p.add_argument("--clip", type=float, default=0.0,
                   help="grad-norm clip; 0 = off (legacy)")
    p.add_argument("--lr_schedule", choices=["const", "cosine"], default="const")
    p.add_argument("--warmup", type=int, default=0,
                   help="linear lr warmup steps (cosine schedule only)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay for eval/best-ckpt weights; 0 = off (legacy)")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="Adam/AdamW beta2 (0.999 = legacy)")
    p.add_argument("--mask_unpredictable", action="store_true",
                   help="mask BCE to predictable cells only (L23/L4; L1 no-op)")
    p.add_argument("--sc_stream", action="store_true",
                   help="E3.5: rejection-sample the TRAINING stream to the SC "
                        "protocol (same filter as the _sc eval corpora); "
                        "L23/L4 only (L1 no-op); default off = legacy stream")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0,
                   help="weight decay; 0 = legacy Adam (bit-identical to history), "
                        ">0 = AdamW decoupled decay (grokking probes)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100,
                   help="train loss/acc log interval (also goes to train.log)")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers for on-the-fly generation (0 = legacy "
                        "synchronous path); data is bit-identical for any worker count")
    p.add_argument("--save_every", type=int, default=0,
                   help="checkpoint every N steps to data/ckpt/<run>.pt "
                        "(0 = save final only)")
    p.add_argument("--evidence_adapter", action="store_true", help="E19D frozen recall plus learned global evidence residual")
    p.add_argument("--init_weights", default=None, help="E19B finalEMA weight initialization; fresh optimizer")
    p.add_argument("--freeze_detectors", action="store_true", help="E19B freeze local detector modules after init_weights")
    p.add_argument("--resume", action="store_true",
                   help="continue from data/ckpt/<run>.pt: exact resume with "
                        "optimizer state, warm-start from legacy model-only ckpts")
    p.add_argument("--bf16", dest="bf16", action="store_true", default=True,
                   help="bf16 autocast for the TRAIN forward+loss on CUDA "
                        "(DEFAULT ON since 2026-09-19, protocol note). "
                        "Measured 2.86x on H200 (art_vanilla d256, batch 512). "
                        "Use --no-bf16 for the fp32 reproduction path "
                        "(bit-identical to the pre-2026-09-19 runs).")
    p.add_argument("--no-bf16", dest="bf16", action="store_false",
                   help="fp32 train forward; bit-identical to history. Needed to "
                        "reproduce or extend a run launched before 2026-09-19.")
    p.add_argument("--bf16-eval", dest="bf16_eval", action="store_true",
                   default=True,
                   help="bf16 autocast for EVAL/rollout forwards (DEFAULT ON "
                        "since 2026-09-19). Rollout is AR (8 steps; 8x49 chain "
                        "levels for diffusion), so this is where eval wall time "
                        "goes. Gate before trusting: scripts/gate_bf16_eval.py "
                        "measures how many sequences change, because SeqAcc is "
                        "exact-match and one flipped cell flips a sequence.")
    p.add_argument("--no-bf16-eval", dest="bf16_eval", action="store_false",
                   help="fp32 eval/rollout. Use this to keep a relaunched arm "
                        "comparable with runs that were already harvesting "
                        "under fp32 eval, so an experiment stays internally "
                        "consistent on one eval precision.")
    p.add_argument("--tf32", action="store_true",
                   help="allow TF32 for fp32 matmuls (default off = highest "
                        "precision, bit-identical to history). 1.42x on H200; "
                        "mutually redundant with --bf16 (bf16 wins).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache_dir", default="data/eval_corpora")
    p.add_argument("--out", default=None)
    p.add_argument("--l1_rule", default="gol", help="'gol' or an integer rule index")
    p.add_argument("--l4_entry", type=int, default=L4_ENTRY_DEFAULT)
    p.add_argument("--l4_default", type=int, default=1)
    p.add_argument("--l4_prior", type=float, default=1.0)
    p.add_argument("--ex_pool", default=None,
                   help="E6.6: pinned exercised-trajectory pool (.npz) mixed into "
                        "every SC training batch (l4v2 + --sc_stream only)")
    p.add_argument("--evidence_loss", choices=["none", "relation", "slot_relation", "slot"], default="none",
                   help="E19 training-only L4B evidence-balanced BCE")
    p.add_argument("--ex_frac", type=float, default=0.0,
                   help="E6.6: fraction of each batch drawn from --ex_pool (0 = off)")
    p.add_argument("--l4v2", default=None, choices=["l4a", "l4b"],
                   help="E6.2/E6.3 constrained rule families (task L4): "
                        "l4a = 4 marginal defaults, l4b = 2 negation pairs. "
                        "Replaces the legacy single-slot bias entirely.")
    # ---- E10 billiard world (external validation of ordered deciding) ----
    p.add_argument("--world", choices=["ca", "billiard"], default="ca",
                   help="E10: 'billiard' dispatches to the self-contained "
                        "billiard train+eval loop (cawm/billiard.py; protocol "
                        "frozen in SETTINGS S12); 'ca' = the legacy CA path")
    p.add_argument("--billiard_arm", choices=["cube", "diag"], default="cube",
                   help="E10: cube = per-frame independent noise levels (swap "
                        "eval of both paths); diag = legacy shared scalar level")
    p.add_argument("--billiard_balls", type=int, default=3)
    p.add_argument("--billiard_collide", action="store_true",
                   help="E10.4: elastic collisions (velocity rotation on "
                        "shared cells); protocol frozen in EXPERIMENT_PLAN "
                        "E10.4 / SETTINGS S12")
    p.add_argument("--billiard_depth", type=int, default=8,
                   help="E10.3: denoiser conv layers (temporal receptive field "
                        "+-depth; 8 covers all 8 history frames from any canvas "
                        "frame). The CA legacy default stays 4.")
    p.add_argument("--billiard_pos_weight", type=float, default=8.0,
                   help="E10: BCE pos_weight for the 1.2%% cell density (S12)")
    p.add_argument("--billiard_eval_n", type=int, default=512)
    args = p.parse_args()
    if args.evidence_loss != "none":
        if not args.sc_stream or (args.evidence_loss != "slot" and not (args.task == "L4" and args.l4v2 == "l4b")):
            p.error("evidence_loss requires sc_stream; relation modes require L4/l4b")
        if args.model not in ("art", "art_kvshift", "constructive", "diffusion"):
            p.error("evidence_loss not supported by this model")
        if args.model == "diffusion" and (not args.frontier or args.chain_nfe):
            p.error("diffusion evidence_loss requires frontier and no chain")
        if args.mask_unpredictable:
            p.error("evidence_loss cannot be combined with mask_unpredictable")


    if args.world == "billiard":
        from .billiard import run_billiard
        run_billiard(args)
        return

    if args.evidence_adapter and (not args.init_weights or args.freeze_detectors):
        p.error("evidence_adapter requires init_weights and freezes all base weights itself")
    if args.init_weights and args.resume:
        p.error("init_weights and resume are distinct mutually exclusive modes")
    if args.freeze_detectors and (not args.init_weights or args.model not in ("art", "art_kvshift")):
        p.error("freeze_detectors requires init_weights and art/art_kvshift")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        assert args.model == "lifegpt" and args.eval_every == 0
        assert args.batch >= world_size
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        args.device = "cuda:" + os.environ["LOCAL_RANK"]
    device = args.device
    # Infra acceleration flags (both default OFF -> fp32 bit-identical to
    # history). Autocast covers the TRAIN forward+loss only; backward, the
    # optimizer, EMA and every eval/rollout path stay fp32, so the reported
    # metrics keep their fp32 numerics and only the weights differ.
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp = bf16_ctx(device, args.bf16)
    if args.model == "constructive":
        run_name = args.out or f"{args.task}_{args.arm}_{args.head}_s{args.seed}"
    else:
        run_name = args.out or f"{args.model}_{args.task}_{args.arm}_s{args.seed}"
    out_dir = os.path.join("data", "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join("data", "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    l4 = None
    rule_override = None
    if args.task == "L1":
        rule_override = R.GOL_RULE if args.l1_rule == "gol" else int(args.l1_rule)
    elif args.task == "L4" and args.l4v2 is None:
        l4 = (args.l4_entry, args.l4_default, args.l4_prior)

    # Model init is RNG-isolated (build_model): DataLoader workers may be
    # spawned in any order relative to construction without shifting init.
    corpora = {}
    assert args.frame_stride == 1 or args.task == "L1", "--frame_stride: L1 only"
    if args.task == "L1":
        rule_label = "gol" if args.l1_rule == "gol" else str(rule_override)
        stag = f"_s{args.frame_stride}" if args.frame_stride != 1 else ""
        corpora["l1_val"] = get_corpus(args.cache_dir,
                                       f"l1_val_{rule_label}{stag}_g{args.grid}",
                                       n=512, master_seed=43, half="train", grid=args.grid,
                                       rule_override=rule_override,
                                       stride=args.frame_stride)
    elif args.task == "L23":
        corpora["l2_val"] = get_corpus(args.cache_dir, f"l2_val_g{args.grid}", n=2048,
                                       master_seed=43, half="train", grid=args.grid)
        # SC variant of the L2 val corpus (train half, rejection sampled):
        # measures L2 on trajectories whose queried slots are all covered.
        corpora["l2_val_sc"] = get_corpus(args.cache_dir, f"l2_val_sc_g{args.grid}",
                                          n=2048, master_seed=43, half="train",
                                          grid=args.grid, self_consistent=True)
        # SC = context self-consistent (rejection sampled, protocol note
        # 2026-08-30): the clean primary. Legacy corpus kept for continuity
        # with historical logs. The older 0.8099 statistic is mean posterior
        # mass on realized futures under an independent-bit prior, not the
        # optimal SeqAcc ceiling on this fixed held-out split.
        corpora["l3_zs"] = get_corpus(args.cache_dir, f"l3_zs_sc_g{args.grid}", n=2048,
                                      master_seed=44, half="zs", grid=args.grid,
                                      self_consistent=True)
        corpora["l3_zs_legacy"] = get_corpus(args.cache_dir, f"l3_zs_g{args.grid}", n=2048,
                                             master_seed=44, half="zs", grid=args.grid)
    elif args.task == "L4" and args.l4v2 is not None:
        # E6.2/E6.3 constrained families: rules drawn from the constrained
        # pool (split decided on the constrained rule id, no preimage leak);
        # predictability = effective coverage under the family constraints.
        corpora["l4_zs"] = get_corpus(args.cache_dir, f"{args.l4v2}_zs_sc_g{args.grid}",
                                      n=2048, master_seed=44, half="zs", grid=args.grid,
                                      self_consistent=True, l4v2=args.l4v2)
        corpora["l2_val"] = get_corpus(args.cache_dir, f"{args.l4v2}_val_sc_g{args.grid}",
                                       n=2048, master_seed=43, half="train", grid=args.grid,
                                       self_consistent=True, l4v2=args.l4v2)
    else:
        corpora["l4_zs"] = get_corpus(args.cache_dir, f"l4_zs_sc_e{args.l4_entry}_p{args.l4_prior}_g{args.grid}",
                                      n=2048, master_seed=44, half="zs", grid=args.grid,
                                      l4=(args.l4_entry, args.l4_default, args.l4_prior),
                                      self_consistent=True, prior_slots=(args.l4_entry,))
        corpora["l4_zs_legacy"] = get_corpus(args.cache_dir, f"l4_zs_e{args.l4_entry}_p{args.l4_prior}_g{args.grid}",
                                             n=2048, master_seed=44, half="zs", grid=args.grid,
                                             l4=(args.l4_entry, args.l4_default, args.l4_prior))
        corpora["l2_val"] = get_corpus(args.cache_dir, f"l4_val_e{args.l4_entry}_p{args.l4_prior}_g{args.grid}",
                                       n=2048, master_seed=43, half="train", grid=args.grid,
                                       l4=(args.l4_entry, args.l4_default, args.l4_prior),
                                       self_consistent=True, prior_slots=(args.l4_entry,))

    model = build_model(args.seed, model=args.model,
                        **model_kwargs(vars(args))).to(device)
    train_model = model
    if world_size > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])])
    initialization = None
    if args.init_weights:
        from .initialization import initialize_and_freeze
        initialization = initialize_and_freeze(model, args.init_weights, args.freeze_detectors)
        if initialization["source_seed"] != args.seed:
            raise ValueError("Initialization seed must match target seed")
    n_train = model.trainable_param_count()
    print(f"[model] {args.model} arm={args.arm} head={args.head} grid={args.grid} "
          f"trainable={n_train} total={model.total_param_count()}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW if args.wd > 0 else torch.optim.Adam)(
        params, lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.wd)
    base_lr = args.lr
    lossf = torch.nn.BCEWithLogitsLoss()
    ema = EMA(model, args.ema) if args.ema > 0 else None
    # L1 has a fixed rule: every slot is in principle predictable from the
    # (known) dynamics, so loss masking is a no-op there.
    use_mask = args.mask_unpredictable and args.task != "L1"
    sc_on = args.sc_stream and args.task != "L1"
    prior_m = np.zeros(R.N_ENTRIES, dtype=bool)
    if args.task == "L4" and args.l4v2 is None:
        prior_m[args.l4_entry] = True
    prior_slots = (args.l4_entry,) if (args.task == "L4" and args.l4v2 is None) else ()
    l4v2 = args.l4v2 if args.task == "L4" else None

    start_step, hist, resumed_from = 0, [], None
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            start_step = int(ck["step"])
            hist = list(ck.get("hist", []))
        else:  # legacy model-only ckpt: warm start, data continues after its steps
            start_step = int(ck["args"]["steps"])
        if ema is not None and "model_ema" in ck:
            ema.shadow = {k: v.to(device) for k, v in ck["model_ema"].items()}
        resumed_from = start_step

    if args.workers > 0:
        chunked = ChunkedStreamDataset(42, half="train", grid=args.grid,
                                       n_chunks=args.steps, chunk_size=args.batch,
                                       l4=l4, rule_override=rule_override,
                                       start_chunk=start_step,
                                       sc_filter=sc_on, prior_slots=prior_slots,
                                       l4v2=l4v2, ex_pool=args.ex_pool,
                                       ex_frac=args.ex_frac,
                                       stride=args.frame_stride)
        loader = torch.utils.data.DataLoader(
            chunked, batch_size=None, num_workers=args.workers,
            persistent_workers=True, prefetch_factor=2)
        batch_iter = iter(loader)
        log_note = (f"workers={args.workers} (chunked stream, bit-identical to sync)"
                    + (" + sc_filter (E3.5)" if sc_on else "")
                    + (f" + ex_pool frac={args.ex_frac} (E6.6)" if chunked.ex_k else ""))
    else:
        assert l4v2 is None or sc_on, \
            "--l4v2 with workers=0 requires --sc_stream (constrained pool " \
            "lives in ChunkedStreamDataset)"
        stream = StreamDataset(42, half="train", grid=args.grid,
                               length=args.steps * args.batch, l4=l4,
                               rule_override=rule_override,
                               stride=args.frame_stride)
        sync_chunked = None
        if sc_on:
            sync_chunked = ChunkedStreamDataset(
                42, half="train", grid=args.grid, n_chunks=args.steps,
                chunk_size=args.batch, l4=l4, rule_override=rule_override,
                start_chunk=start_step, sc_filter=True, prior_slots=prior_slots,
                l4v2=l4v2, ex_pool=args.ex_pool, ex_frac=args.ex_frac,
                stride=args.frame_stride)
        batch_iter = None
        log_note = ("workers=0 (legacy synchronous)"
                    + (" + sc_filter (E3.5)" if sc_on else ""))

    t0 = time.time()
    log_path = os.path.join(out_dir, "train.log")
    eval_on = args.eval_every > 0  # 0 = no inline eval (watch_eval.py does it
    # in a parallel process off the saved ckpt; protocol note 2026-08-29)

    def log(msg):
        if rank != 0:
            return
        print(msg, flush=True)
        with open(log_path, "a") as logf:
            logf.write(msg + "\n")

    fingerprint = code_fingerprint()
    log(f"[run] {run_name} -> {out_dir}")
    if resumed_from is not None:
        log(f"[resume] continuing from step {resumed_from} "
            f"(ckpt: {ckpt_path})")
    log(f"[code] cawm sha256:{fingerprint}")
    log(f"[data] {log_note}")
    log(f"[opt] lr={args.lr} wd={args.wd} {'adamw' if args.wd > 0 else 'adam'}")
    if args.clip > 0 or args.lr_schedule != "const" or args.ema > 0 \
            or args.beta2 != 0.999 or args.mask_unpredictable or args.sc_stream:
        log(f"[recipe-v2] clip={args.clip} lr_schedule={args.lr_schedule} "
            f"warmup={args.warmup} ema={args.ema} beta2={args.beta2} "
            f"mask_unpredictable={args.mask_unpredictable} "
            f"sc_stream={args.sc_stream}")
    if args.bf16 or args.bf16_eval or args.tf32:
        log(f"[infra] bf16_train={args.bf16} bf16_eval={args.bf16_eval} "
            f"tf32={args.tf32} (bf16_eval governs every eval/rollout forward, "
            f"inline and in watch_eval; both default ON since 2026-09-19)")
    if args.bf16_eval and args.model == "diffusion" and not args.no_cascade:
        log("[infra][WARN] bf16_eval is ON for a cascaded diffusion arm. "
            "Measured 2026-09-19 (scripts/gate_bf16_eval.py, "
            "diffF_L23_frontier_s42_bf16, n=128): bf16 eval moves ~50% of "
            "sequences and shifts noCanvas by up to 4.7pp, because rounding "
            "compounds over the 49-level reverse chain and flips the final "
            "sign(). SC SeqAcc is far more stable (0 to -1.6pp). Do NOT "
            "compare a bf16-evaluated diffusion noCanvas against an "
            "fp32-evaluated one: pass --no-bf16-eval to match siblings.")
    if args.chain_nfe > 0:
        log(f"[chain] nfe={args.chain_nfe} init={args.chain_init} "
            f"aux={args.chain_aux} (E3.8: differentiable reverse chain, "
            f"train=inference, no ground-truth leak in the canvas)")
    if args.frontier:
        log("[frontier] E3.9: teacher-forced prefix + frontier-only BCE; "
            "eval auto-routes to frame_ar+commit")

    best = {"metric": -1.0, "step": 0}
    best_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

    def do_eval(step, tag):
        """Run eval under the EMA swap (if active); track + save best-ckpt."""
        ctx = ema.swap(model) if ema is not None else contextlib.nullcontext()
        with ctx:
            results = run_eval(model, corpora, device, tag,
                               l4_entry=(args.l4_entry if args.task == "L4"
                                         and args.l4v2 is None else None),
                               l4v2=l4v2, bf16=args.bf16_eval,
                               cov_bins=(args.task == "L23"), log=log)
            pm = primary_metric(results, args.task)
            if pm is not None and pm > best["metric"]:
                best["metric"], best["step"] = pm, step
                save_best(best_path, model, step, results)
                log(f"[best] step {step}: primary {pm:.4f} -> {best_path}")
        model.train()
        return results

    model.train()
    if world_size > 1:
        # Identical init, independent FCM draws on disjoint world shards.
        torch.manual_seed(args.seed + rank)
    for step in range(start_step + 1, args.steps + 1):
        if batch_iter is not None:
            batch = next(batch_iter)
        elif sc_on:
            batch = sync_chunked[step - 1 - start_step]
        else:
            idx = list(range((step - 1) * args.batch, step * args.batch))
            batch = stream.get_batch(idx)
        frames = batch["frames"].to(device).float()              # (B,16,H,W) 0/1
        x = 2 * frames - 1
        targets = frames[:, PREFIX_LEN:]
        loss_mask = None
        if use_mask:
            covs_np = batch["cov"].numpy()
            if l4v2 is not None:
                covs_np = R.l4v2_effective_coverage(covs_np, l4v2)
            np_mask = unpredictable_cells_mask(
                batch["frames"].numpy(), covs_np, prior_m)
            loss_mask = torch.from_numpy(np_mask).to(device)   # (B,8,H,W) bool
        evidence = None
        if args.evidence_loss != "none":
            evidence = evidence_groups(frames, batch["cov"].to(device), args.evidence_loss)
        opt.zero_grad()
        if args.model == "lifegpt":
            if world_size > 1:
                start = args.batch * rank // world_size
                end = args.batch * (rank + 1) // world_size
                x = x[start:end]
                targets = targets[start:end]
            micro = args.microbatch or args.batch
            total_loss, total_correct = 0.0, 0.0
            for lo in range(0, len(x), micro):
                hi = min(lo + micro, len(x))
                sync = (train_model.no_sync() if world_size > 1 and hi < len(x)
                        else nullcontext())
                with sync:
                    with amp:
                        logits = train_model(x[lo:hi].unsqueeze(1))
                        part_loss = lossf(logits, targets[lo:hi])
                    if not torch.isfinite(part_loss):
                        raise FloatingPointError(f"nonfinite LifeGPT loss at step {step}")
                    # DDP averages gradients over ranks. Weight each shard by
                    # its world count so uneven shards still give batch512's mean.
                    weight = (hi-lo) * world_size / args.batch
                    (part_loss * weight).backward()
                total_loss += part_loss.detach().item() * ((hi-lo) / len(x))
                total_correct += ((logits.detach() > 0) == targets[lo:hi]).sum().item()
            loss = torch.tensor(total_loss, device=device)
            acc = total_correct / targets.numel()
            if world_size > 1:
                stats = torch.tensor([total_loss * len(x), total_correct], device=device)
                dist.all_reduce(stats)
                loss = stats[0] / args.batch
                acc = (stats[1] / (args.batch * targets[0].numel())).item()
        else:
            with amp:
                if args.model in ("diffusion", "diffusion_vanilla", "diffnca_cfg"):
                    # single-step denoising: stratified t, closed-form noising,
                    # BCE-with-logits on the clean frame (docs/SETTINGS.md §6.5)
                    if args.t_frame_slope > 0 and hasattr(model, "t_frame_slope_now"):
                        if args.t_frame_slope_anneal > 0:
                            model.t_frame_slope_now = args.t_frame_slope * max(
                                0.0, 1.0 - step / args.t_frame_slope_anneal)
                        else:
                            model.t_frame_slope_now = args.t_frame_slope
                    if getattr(model, "chain_nfe", None):
                        # E3.8: differentiable reverse chain (final-level BCE +
                        # per-step aux); the chain canvas holds the model's own
                        # belief, never the ground truth (no leak; train=inference)
                        loss, acc_t = model.chain_loss(x.unsqueeze(1),
                                                       aux_weight=args.chain_aux)
                    elif getattr(model, "frontier", False):
                        # E3.9: teacher-forced prefix + frontier-only BCE, matched
                        # to frame_ar+commit inference
                        loss, acc_t = model.frontier_loss(x.unsqueeze(1), evidence_groups=evidence,
                                                          evidence_mode=args.evidence_loss)
                    else:
                        loss, acc_t = model.training_loss(x.unsqueeze(1),
                                                          loss_mask=loss_mask)
                    acc = acc_t.item()
                else:
                    logits = model(x.unsqueeze(1))
                    if evidence is not None:
                        loss = evidence_bce(logits, targets, evidence, args.evidence_loss)
                    elif loss_mask is not None:
                        bce = F.binary_cross_entropy_with_logits(
                            logits, targets, reduction="none")
                        keep = ~loss_mask
                        loss = (bce * keep).sum() / keep.sum().clamp(min=1)
                    else:
                        loss = lossf(logits, targets)
        if args.model != "lifegpt":
            loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        # lr schedule: mult applied to the base lr each step
        if args.lr_schedule != "const":
            mult = lr_mult(step, args.lr_schedule, args.warmup, args.steps)
            for g in opt.param_groups:
                g["lr"] = base_lr * mult
        opt.step()
        if ema is not None:
            ema.update(model)
        if step % args.log_every == 0 or step == 1 or step == start_step + 1:
            if args.model not in ("lifegpt", "diffusion", "diffusion_vanilla", "diffnca_cfg"):
                acc = ((logits > 0).to(torch.uint8) == targets).float().mean().item()
            log(f"step {step:5d}  loss {loss.item():.4f}  "
                f"tf-pixel {acc:.4f}  ({time.time()-t0:.0f}s)")
            hist.append({"step": step, "loss": float(loss.item()), "tf_pixel": acc})
        if eval_on and (step % args.eval_every == 0 or step == args.steps):
            do_eval(step, f"step {step}")
        if rank == 0 and args.save_every and step % args.save_every == 0:
            save_ckpt(ckpt_path, model, opt, step, hist, args, ema=ema)

    if world_size > 1:
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return
    final = do_eval(args.steps, "final") if eval_on else None
    save_ckpt(ckpt_path, model, opt, args.steps, hist, args, ema=ema)
    summary = {
        "run": run_name, "task": args.task, "model": args.model,
        "arm": args.arm, "head": args.head, "grid": args.grid,
        "seed": args.seed, "steps": args.steps, "batch": args.batch,
        "initialization": initialization,
        "evidence_loss": args.evidence_loss,
        "workers": args.workers, "log_every": args.log_every,
        "microbatch": args.microbatch,
        "world_size": world_size,
        "peak_memory_bytes": (torch.cuda.max_memory_allocated(device)
                              if str(device).startswith("cuda") else 0),
        "eval_every": args.eval_every, "lr": args.lr, "wd": args.wd,
        "save_every": args.save_every, "resumed_from": resumed_from,
        "eval_mode": "inline" if eval_on else "external-watch",
        "trainable_params": n_train, "total_params": model.total_param_count(),
        "code_sha256": fingerprint,
        "wall_time_s": time.time() - t0, "final": final, "hist": hist,
        "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                   if str(device).startswith("cuda") else None),
        "best": best, "recipe_v2": {
            "clip": args.clip, "lr_schedule": args.lr_schedule,
            "warmup": args.warmup, "ema": args.ema, "beta2": args.beta2,
            "mask_unpredictable": args.mask_unpredictable,
            "sc_stream": args.sc_stream,
        },
        "chain": {"nfe": args.chain_nfe, "init": args.chain_init,
                  "aux": args.chain_aux},
        "infra": {"bf16_train": args.bf16, "bf16_eval": args.bf16_eval,
                  "tf32": args.tf32},
        "frontier": args.frontier,
        "l4v2": args.l4v2,
        "protocol": "post-submission regenerated (seed=42 split), hash-split protocol",
    }
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[done] {run_name}: {time.time()-t0:.0f}s, saved {out_dir}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
