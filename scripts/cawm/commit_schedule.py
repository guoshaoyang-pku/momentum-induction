'''Eval-only ordered commit scheduling for CAWM per-frame diffusion denoisers.

Bounded extension: no training code or model classes are touched, there are no
trainable parameters, and ground truth is never consulted.  The uniform joint
sampler baseline stays in ``model.rollout`` (nfe=8); this module only adds
ordered per-frame commit selection on top of a trained plain denoiser exposing
``denoise(x, hist, levels) -> (logits, logits)`` with per-frame noise levels
(``t_per_frame=True``), spins at scale +-1.

Per round, ONLY incomplete worlds are batched through exactly one denoiser
call.  Committed frames enter as hard +-1 spins at level 0; pending frames
hold their ORIGINAL random noise canvas at level ``T_STEPS - 1``.  The same
logits supply both the confidence score -- mean over cells of
``|tanh(logits / 2)|`` -- and the committed predictions; there is no separate,
uncounted scoring pass.  All selections of a round are computed before any
commit of that round, so parallel mode never sees same-round commits.
Hard-frozen predictions and states never change afterwards.

Determinism: the only randomness is the initial noise draw (global RNG, the
caller controls ``torch.manual_seed``; or pass ``initial_noise``) and the
independent generator used by ``mode='random'`` (``order_seed``).
'''

import torch

from .models.diffusion import T_STEPS

__all__ = ['commit_rollout']

_MODES = ('causal', 'confident', 'random', 'parallel')
_N_FRAMES = 8
_PENDING_LEVEL = T_STEPS - 1


def _first_true(mask):
    '''Index of the first True in each row of a (b, F) bool mask.

    Every row is assumed to contain at least one True (active worlds always
    have at least one pending frame).
    '''
    n = mask.shape[1]
    arange = torch.arange(n, device=mask.device).expand_as(mask)
    filled = torch.where(mask, arange, torch.full_like(arange, n))
    return filled.min(dim=1).values


def _one_hot_bool(index, n):
    return torch.nn.functional.one_hot(index, num_classes=n).bool()


@torch.no_grad()
def commit_rollout(model, prefix_pm1, mode, threshold=0.99, initial_noise=None, order_seed=0):
    '''Roll out the 8 target frames under an ordered commit schedule.

    Args:
        model: trained denoiser with ``t_per_frame=True`` exposing
            ``denoise(x, hist, levels) -> (logits, logits)`` where ``x`` and
            ``hist`` are (B, 8, H, W) and ``levels`` is integer (B, 8).
        prefix_pm1: (B, 1, 8, H, W) history tensor of +-1 spins.
        mode: 'causal' (smallest pending index), 'confident' (max pending
            confidence, tie -> smallest index), 'random' (max independent
            random score among pending), or 'parallel' (all pending frames
            with confidence >= ``threshold``; if none pass, the single
            highest-confidence pending frame, tie -> earliest, so progress
            is guaranteed).
        threshold: parallel-mode confidence gate, must lie in (0, 1].
        initial_noise: optional (B, 8, H, W) noise canvas.  When None it is
            drawn once via ``torch.randn`` from the global RNG (the caller
            controls ``torch.manual_seed`` beforehand).
        order_seed: seed for the independent generator of ``mode='random'``.

    Returns:
        preds: uint8 (B, 8, H, W), committed predictions (1 = spin up).
        commit_round: int64 (B, 8), zero-based round index at which each
            frame was committed (for auditing batches).
        calls_per_world: int64 (B,), number of denoiser calls whose batch
            included each world (incremented for active worlds only).
        actual_forward_calls: int, actual number of ``model.denoise`` calls.
    '''
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}, got {mode!r}')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError('threshold must be a real number in (0, 1]')
    thr = float(threshold)
    if not 0.0 < thr <= 1.0:
        raise ValueError(f'threshold must be in (0, 1], got {thr}')
    if not isinstance(prefix_pm1, torch.Tensor):
        raise ValueError('prefix_pm1 must be a torch.Tensor')
    if prefix_pm1.ndim != 5 or prefix_pm1.shape[1] != 1 or prefix_pm1.shape[2] != _N_FRAMES:
        raise ValueError(
            f'prefix_pm1 must have shape (B, 1, {_N_FRAMES}, H, W), got {tuple(prefix_pm1.shape)}')
    if not bool(torch.all(prefix_pm1.abs() == 1)):
        raise ValueError('prefix_pm1 must contain only +-1 spins')
    if not bool(getattr(model, 't_per_frame', False)):
        raise ValueError('commit_rollout only supports models trained with t_per_frame=True')

    batch, _, n_frames, height, width = prefix_pm1.shape
    device = prefix_pm1.device
    dtype = prefix_pm1.dtype if prefix_pm1.is_floating_point() else torch.float32
    hist = prefix_pm1[:, 0].to(dtype)

    if initial_noise is None:
        noise = torch.randn(batch, n_frames, height, width, device=device, dtype=dtype)
    else:
        if not isinstance(initial_noise, torch.Tensor) or tuple(initial_noise.shape) != (
                batch, n_frames, height, width):
            raise ValueError(
                f'initial_noise must have shape ({batch}, {n_frames}, {height}, {width})')
        noise = initial_noise.to(device=device, dtype=dtype).clone()

    state = noise.clone()  # pending frames keep their original noise forever
    committed = torch.zeros(batch, n_frames, dtype=torch.bool, device=device)
    preds = torch.zeros(batch, n_frames, height, width, dtype=torch.uint8, device=device)
    commit_round = torch.full((batch, n_frames), -1, dtype=torch.int64, device=device)
    calls_per_world = torch.zeros(batch, dtype=torch.int64, device=device)
    actual_forward_calls = 0

    order_gen = None
    if mode == 'random':
        order_gen = torch.Generator(device='cpu')
        order_gen.manual_seed(int(order_seed))

    # Every round commits at least one frame per active world, so at most
    # n_frames rounds are ever needed.
    for round_idx in range(n_frames):
        if bool(committed.all()):
            break
        idx = (~committed).any(dim=1).nonzero(as_tuple=True)[0]
        x = state[idx]
        pending = ~committed[idx]
        levels = pending.to(torch.int64) * _PENDING_LEVEL  # committed -> level 0

        out = model.denoise(x, hist[idx], levels)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        actual_forward_calls += 1
        calls_per_world[idx] += 1

        # The scoring logits ARE the prediction logits: one counted pass.
        conf = torch.tanh(logits / 2.0).abs().mean(dim=(-2, -1))

        # All selections are fixed BEFORE any commit of this round.
        if mode == 'causal':
            sel = _one_hot_bool(_first_true(pending), n_frames)
        elif mode == 'confident':
            score = conf.masked_fill(~pending, float('-inf'))
            best = score.max(dim=1, keepdim=True).values
            sel = _one_hot_bool(_first_true(pending & (score == best)), n_frames)
        elif mode == 'random':
            draw = torch.rand(batch, n_frames, generator=order_gen)
            score = draw[idx.cpu()].to(device=device).masked_fill(~pending, float('-inf'))
            best = score.max(dim=1, keepdim=True).values
            sel = _one_hot_bool(_first_true(pending & (score == best)), n_frames)
        else:  # parallel
            sel = pending & (conf >= thr)
            missing = ~sel.any(dim=1)
            if bool(missing.any()):
                # Guarantee progress: highest-confidence pending, tie earliest.
                score = conf.masked_fill(~pending, float('-inf'))
                best = score.max(dim=1, keepdim=True).values
                fallback = _one_hot_bool(_first_true(pending & (score == best)), n_frames)
                sel = torch.where(missing[:, None], fallback, sel)

        sel4 = sel[:, :, None, None]
        hard = torch.where(logits > 0, torch.ones_like(x), -torch.ones_like(x))
        state[idx] = torch.where(sel4, hard, x)
        preds[idx] = torch.where(sel4, (logits > 0).to(torch.uint8), preds[idx])
        rounds = commit_round[idx]
        rounds[sel] = round_idx
        commit_round[idx] = rounds
        committed[idx] = committed[idx] | sel

    assert bool(committed.all()), 'internal error: schedule failed to complete'
    return preds, commit_round, calls_per_world, actual_forward_calls
