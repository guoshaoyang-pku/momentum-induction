"""Vectorized outer-totalistic CA simulator (wrap boundary, 0/1 storage)."""

import numpy as np

from .rules import N_ENTRIES

_SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def rule_tables(rules):
    """(B,) rule indices -> (B, 18) uint8 truth tables; table[:, i] = bit i."""
    rules = np.asarray(rules, dtype=np.int64)
    return ((rules[:, None] >> np.arange(N_ENTRIES)[None, :]) & 1).astype(np.uint8)


def neighbor_count(x):
    """Moore-8 live-neighbor count with wrap boundary. x: (B, H, W) in {0,1}."""
    out = np.zeros(x.shape, dtype=np.int16)
    for dy, dx in _SHIFTS:
        out += np.roll(x, (dy, dx), axis=(1, 2))
    return out


def ca_step(x, tables):
    """One CA step. x: (B, H, W) uint8; tables: (B, 18) uint8."""
    n = neighbor_count(x)
    idx = (9 * x.astype(np.int16) + n).astype(np.int64)  # entry = 9*s + n
    b = np.arange(x.shape[0])
    return tables[b[:, None, None], idx]


def simulate(rules, inits, steps):
    """Simulate `steps` transitions. Returns (B, steps+1, H, W) uint8 frames."""
    tables = rule_tables(rules)
    x = np.asarray(inits, dtype=np.uint8)
    frames = [x]
    for _ in range(steps):
        x = ca_step(x, tables)
        frames.append(x)
    return np.stack(frames, axis=1)


def to_pm1(x):
    return 2 * np.asarray(x, dtype=np.float32) - 1


def to_01(x):
    return np.asarray(x, dtype=np.float32)
