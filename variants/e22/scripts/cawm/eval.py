"""Rollout evaluation and metrics (SeqAcc primary, stratified by coverage)."""

import numpy as np
import torch

from .data import PREFIX_LEN


def bf16_ctx(device, enabled):
    """Single source of truth for forward-pass precision.

    bf16 autocast on CUDA when `enabled`, a no-op on CPU or when disabled.
    train.py's inline eval, watch_eval.py and the bf16 gate all route through
    here, so a harvested number cannot depend on which process produced it.
    """
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=bool(enabled) and str(device).startswith("cuda"))


def rollout_corpus(model, corpus, device="cpu", batch=256, bf16=False, **kw):
    """Run 8-step AR rollouts over a pinned corpus. Returns preds (N,8,H,W) uint8.
    Extra kwargs (e.g. ablate_canvas) are forwarded to model.rollout.
    `bf16` is eval precision, NOT forwarded to the model."""
    model.eval()
    frames = corpus["frames"]
    N = frames.shape[0]
    preds = []
    with torch.no_grad(), bf16_ctx(device, bf16):
        for i in range(0, N, batch):
            fr = torch.from_numpy(frames[i:i + batch].astype(np.float32)).to(device)
            prefix = (2 * fr[:, :PREFIX_LEN] - 1).unsqueeze(1)  # (B,1,8,H,W)
            preds.append(model.rollout(prefix, **kw).cpu().numpy())
    return np.concatenate(preds, axis=0)


def metrics(preds, corpus):
    """preds: (N,8,H,W) uint8 vs corpus targets frames[:, 8:16]."""
    targets = corpus["frames"][:, PREFIX_LEN:]
    cov = corpus["covs"]
    exact = (preds == targets).all(axis=(1, 2, 3))
    pixel = (preds == targets).mean()
    out = {
        "seq_acc": float(exact.mean()),
        "pixel_acc": float(pixel),
        "n": int(len(exact)),
        "full_cov_frac": float(cov.all(axis=1).mean()),
    }
    full = cov.all(axis=1)
    if full.any():
        out["seq_acc_full_cov"] = float(exact[full].mean())
    if (~full).any():
        out["seq_acc_partial_cov"] = float(exact[~full].mean())
    return out


def metrics_cov_bins(preds, corpus):
    """SeqAcc binned by number of covered prefix slots k (out of N_ENTRIES).

    The dilution decomposition: accuracy on rules whose prefix reveals the
    whole truth table (k=18) vs rules with partial evidence (k<18). Returned
    dict maps k -> {n, seq_acc}; JSON keys become strings on dump.
    """
    targets = corpus["frames"][:, PREFIX_LEN:]
    exact = (preds == targets).all(axis=(1, 2, 3))
    k = corpus["covs"].sum(axis=1)
    out = {"mean_cov": float(k.mean())}
    for kk in np.unique(k):
        m = k == kk
        out[int(kk)] = {"n": int(m.sum()), "seq_acc": float(exact[m].mean())}
    return out


def metrics_stratified_slot(preds, corpus, entry):
    """SeqAcc split by whether slot `entry` is covered in the prefix (L4)."""
    targets = corpus["frames"][:, PREFIX_LEN:]
    exact = (preds == targets).all(axis=(1, 2, 3))
    covered = corpus["covs"][:, entry]
    out = {}
    if covered.any():
        out["seq_acc_slot_covered"] = float(exact[covered].mean())
    if (~covered).any():
        out["seq_acc_slot_uncovered"] = float(exact[~covered].mean())
    out["slot_uncovered_frac"] = float((~covered).mean())
    return out


def metrics_stratified_v2(preds, corpus, cfg_name):
    """E6.2/E6.3 strata for the constrained families (L4-v2).

    default-exercised: some default slot is queried but uncovered — the
    model had to use the family default. derived-exercised: some pair slot
    is queried, uncovered, with its partner covered — the model had to do
    the one-hop inference (read partner, negate). SC corpora guarantee
    every queried slot is predictable, so these strata partition the
    "constraint actually used" trajectories."""
    from . import rules as R
    from .data import queried_slots_mask
    cfg = R.L4V2_CONFIGS[cfg_name]
    targets = corpus["frames"][:, PREFIX_LEN:]
    exact = (preds == targets).all(axis=(1, 2, 3))
    covs = corpus["covs"]
    queried = queried_slots_mask(corpus["frames"])
    out = {}
    if cfg["defaults"]:
        d = np.zeros(len(exact), dtype=bool)
        for e, _ in cfg["defaults"]:
            d |= queried[:, e] & ~covs[:, e]
        out["default_exercised_frac"] = float(d.mean())
        if d.any():
            out["seq_acc_default_exercised"] = float(exact[d].mean())
        if (~d).any():
            out["seq_acc_default_idle"] = float(exact[~d].mean())
    if cfg["pairs"]:
        v = np.zeros(len(exact), dtype=bool)
        for a, b in cfg["pairs"]:
            v |= queried[:, b] & ~covs[:, b] & covs[:, a]
            v |= queried[:, a] & ~covs[:, a] & covs[:, b]
        out["derived_exercised_frac"] = float(v.mean())
        if v.any():
            out["seq_acc_derived_exercised"] = float(exact[v].mean())
        if (~v).any():
            out["seq_acc_derived_idle"] = float(exact[~v].mean())
    return out


def _tf_forward(model, frames_pm1):
    """Teacher-forced logits (B,8,H,W) from TRUE inputs (B,1,16,H,W)."""
    try:
        return model(frames_pm1)
    except NotImplementedError:
        pass
    # Denoisers (DiffusionVanilla et al.): one-step denoise of the clean
    # target frame at the model's own lowest training noise level, given the
    # TRUE 8-frame history window (the diffusion analogue of teacher forcing).
    denoise = getattr(model, "denoise", None)
    if denoise is not None:
        from .models.diffusion import T_STEPS, alpha_t, sigma_t
        t_min = T_STEPS - 1
        outs = []
        B = frames_pm1.shape[0]
        dev = frames_pm1.device
        for j in range(8):
            hist = frames_pm1[:, 0, j:j + 8]                 # (B,8,H,W) +-1
            x0 = frames_pm1[:, 0, j + 8]                     # (B,H,W) +-1
            t = torch.full((B,), t_min, dtype=torch.long, device=dev)
            xt = alpha_t(t).view(B, 1, 1) * x0 \
                + sigma_t(t).view(B, 1, 1) * torch.randn_like(x0)
            u, _ = denoise(xt, hist, t)
            outs.append(u)
        return torch.stack(outs, dim=1)                      # (B,8,H,W) logits
    step = getattr(model, "_step", None)
    if step is None:
        raise NotImplementedError(
            "teacher_forced_corpus: model exposes neither a teacher-forced "
            "forward(frames_pm1) -> (B,8,H,W) logits nor a per-step "
            "_step(window_pm1); cannot compute one-step teacher-forced "
            "accuracy for this model."
        )
    return torch.stack(
        [step(frames_pm1[:, :, j:j + 8]) for j in range(8)], dim=1
    )


def teacher_forced_corpus(model, corpus, device="cpu", batch=256, bf16=False):
    """One-step teacher-forced per-cell accuracy over a pinned corpus.

    For each trajectory and each j in 0..7: feed frames[j:j+8] (TRUE history)
    and compare the model's prediction for frame 8+j against the true target
    frame. Returns dict:
      {"tf_pixel_acc": mean over (trajectory, j, cell) of equality,
       "tf_seq_acc":  fraction of (trajectory, j) pairs with ALL 64 cells correct,
       "n": number of trajectories, "n_steps": 8}
    This is the convention LifeGPT ("99.9% of predicted cells") and
    AutomataGPT ("98.5% perfect one-step forecasts") report."""
    frames = np.asarray(corpus["frames"])
    n = frames.shape[0]
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    correct_cells = 0
    total_cells = 0
    perfect_pairs = 0
    total_pairs = 0
    with torch.no_grad():
        for i in range(0, n, batch):
            f = torch.as_tensor(frames[i:i + batch], device=device)
            frames_pm1 = (2.0 * f.float() - 1.0).unsqueeze(1)
            with bf16_ctx(device, bf16):
                logits = _tf_forward(model, frames_pm1)
            pred = (logits.float() > 0)
            tgt = f[:, 8:16].bool()
            eq = (pred == tgt)
            correct_cells += int(eq.sum().item())
            total_cells += int(eq.numel())
            eq_flat = eq.reshape(eq.shape[0], 8, -1)
            perfect_pairs += int(eq_flat.all(dim=-1).sum().item())
            total_pairs += eq.shape[0] * 8
    if was_training and hasattr(model, "train"):
        model.train()
    return {
        "tf_pixel_acc": float(correct_cells / max(total_cells, 1)),
        "tf_seq_acc": float(perfect_pairs / max(total_pairs, 1)),
        "n": int(n),
        "n_steps": 8,
    }
