"""Vectorized batch CA simulation + SC/exercised filtering for pool builds.

Generates N trajectories at once (per-sample rule tables), then computes
coverage/predictability/exercised masks as boolean matrices. ~150x faster
than the per-trajectory path, which makes exercised-pool construction
(the l4b exercised rate is ~0.02%) practical.
"""
import numpy as np

from . import ca as ca_mod


def batch_trajectories(tables: np.ndarray, rng, steps=16, p_lo=0.3, p_hi=0.0):
    """tables (N,18) uint8 -> states (N,steps,8,8) uint8. p per-sample
    ~ U(p_lo, p_hi) if p_hi > 0 else fixed p_lo."""
    N = tables.shape[0]
    p = rng.uniform(p_lo, p_hi if p_hi > 0 else p_lo, size=(N, 1, 1))
    g = (rng.random((N, 8, 8)) < p).astype(np.uint8)
    out = np.empty((N, steps, 8, 8), np.uint8)
    out[:, 0] = g
    for t in range(steps - 1):
        padded = np.zeros((N, 10, 10), np.uint8)
        padded[:, 1:-1, 1:-1] = g
        n = sum(padded[:, dy:dy + 8, dx:dx + 8]
                for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1))
        slots = g * 9 + n
        g = np.take_along_axis(tables[:, None, None, :],
                               slots[..., None], axis=3)[..., 0]
        out[:, t + 1] = g
    return out


def batch_slot_coverage(states: np.ndarray):
    """states (N,T,8,8) -> covered (N,T-1,18) bool: slot queried at each
    transition (slot of the SOURCE grid)."""
    N, T = states.shape[:2]
    padded = np.zeros((N, T, 10, 10), np.uint8)
    padded[:, :, 1:-1, 1:-1] = states
    n = sum(padded[:, :, dy:dy + 8, dx:dx + 8]
            for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1))
    slots = states * 9 + n                       # (N,T,8,8)
    covered = np.zeros((N, T - 1, 18), bool)
    for sl in range(18):
        covered[:, :, sl] = (slots[:, :-1] == sl).any(axis=(2, 3))
    return covered, slots


def batch_sc_exercised(states: np.ndarray, cfg_name: str = None):
    """-> (sc (N,) bool, exercised (N,) bool). History = transitions
    0..6 (source grids 0..6), queries = transitions 7..14."""
    covered, _ = batch_slot_coverage(states)
    seen = covered[:, :7].any(axis=1)            # (N,18)
    q = covered[:, 7:].any(axis=1)               # (N,18) queried in future
    if cfg_name is None:
        sc = (q <= seen).all(axis=1)
        return sc, np.zeros(states.shape[0], bool)
    cfg = ca_mod.L4_CONFIGS[cfg_name]
    pred = seen.copy()
    for e, _ in cfg["defaults"]:
        pred[:, e] = True
    for a, b in cfg["pairs"]:
        either = seen[:, a] | seen[:, b]
        pred[:, a] = either
        pred[:, b] = either
    sc = (q <= pred).all(axis=1)
    ex = np.zeros(states.shape[0], bool)
    for e, _ in cfg["defaults"]:
        ex |= q[:, e] & ~seen[:, e]
    for a, b in cfg["pairs"]:
        ex |= q[:, b] & ~seen[:, b] & seen[:, a]
        ex |= q[:, a] & ~seen[:, a] & seen[:, b]
    return sc, ex


def draw_exercised_pool(cfg_name, n_target, half, seed, p_lo=0.3, p_hi=0.0,
                        batch=4096, max_draws=10**9, rng_out=False):
    """Sample random family rules from `half`, keep SC+exercised
    trajectories. Returns dict with states (M,16,8,8) and rules (M,)."""
    rng = np.random.default_rng(seed)
    keep_states, keep_rules = [], []
    drawn = 0
    while sum(len(k) for k in keep_rules) < n_target and drawn < max_draws:
        rules = rng.integers(0, 1 << 18, size=batch)
        ok = np.array([ca_mod.rule_satisfies(int(r), cfg_name)
                       and (ca_mod.is_train_rule(int(r)) == (half == "train"))
                       for r in rules])
        rules = rules[ok]
        if len(rules) == 0:
            continue
        tables = np.stack([ca_mod.table_from_rule(int(r)) for r in rules])
        states = batch_trajectories(tables, rng, 16, p_lo, p_hi)
        drawn += len(rules)
        sc, ex = batch_sc_exercised(states, cfg_name)
        m = sc & ex
        if m.any():
            keep_states.append(states[m])
            keep_rules.append(rules[m])
    states = np.concatenate(keep_states) if keep_states else \
        np.empty((0, 16, 8, 8), np.uint8)
    rules = np.concatenate(keep_rules) if keep_rules else \
        np.empty((0,), np.int64)
    out = {"states": states[:n_target], "rules": rules[:n_target],
           "drawn": drawn}
    return out
