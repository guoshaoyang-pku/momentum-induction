"""Zero-boundary totalistic 2-state CA on a small arena (pixel lab track).

The mainline uses a toroidal arena. Here cells outside the arena count as
white (0), matching the video rendering where the arena sits in a uniform
background: a white cell and the background are pixel-identical.

A rule is an 18-bit table indexed by slot = own_state*9 + black_neighbours.
"""
import numpy as np

GRID = 8
N_SLOTS = 18


def table_from_rule(rule: int) -> np.ndarray:
    return np.array([(rule >> slot) & 1 for slot in range(N_SLOTS)], np.uint8)


def rule_from_parts(birth, survive) -> int:
    r = 0
    for n in birth:
        r |= 1 << n
    for n in survive:
        r |= 1 << (9 + n)
    return r


RULE_GOL = rule_from_parts({3}, {2, 3})
RULE_B2S3 = rule_from_parts({2}, {3})
RULE_HIGHLIFE = rule_from_parts({3, 6}, {2, 3})


def is_train_rule(rule: int) -> bool:
    return ((rule * 2654435761) >> 31) & 1 == 0


def _neighbour_count(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    p = np.zeros((H + 2, W + 2), np.uint8)
    p[1:-1, 1:-1] = grid
    n = np.zeros((H, W), np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            n += p[dy:dy + H, dx:dx + W]
    return n


def step(grid: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table[grid * 9 + _neighbour_count(grid)]


def trajectory(table, rng, steps=16, p_init=0.3, grid=GRID):
    g = (rng.random((grid, grid)) < p_init).astype(np.uint8)
    out = [g]
    for _ in range(steps - 1):
        g = step(g, table)
        out.append(g)
    return np.stack(out)


def slots_of(grid_prev: np.ndarray) -> np.ndarray:
    return grid_prev * 9 + _neighbour_count(grid_prev)


# L4 family constraints, ported from the mainline (cawm/rules.py L4V2_CONFIGS
# + legacy single-slot L4). Membership is decided on the CONSTRAINED rule id
# itself, then hash-split, so train/ZS families are truly disjoint.
L4_CONFIGS = {
    "l4":  {"defaults": ((9, 1),), "pairs": ()},
    "l4a": {"defaults": ((2, 1), (9, 1), (6, 0), (16, 0)), "pairs": ()},
    "l4b": {"defaults": (), "pairs": ((4, 13), (3, 12))},
}


def rule_satisfies(rule: int, cfg_name: str) -> bool:
    cfg = L4_CONFIGS[cfg_name]
    for e, v in cfg["defaults"]:
        if ((rule >> e) & 1) != v:
            return False
    for a, b in cfg["pairs"]:
        if ((rule >> a) & 1) == ((rule >> b) & 1):
            return False
    return True


def _sc_query_check(query_slots, seen, cfg_name=None):
    """Is every future-queried slot predictable? With an L4 family cfg,
    default slots are always predictable and a pair slot is predictable iff
    either member is evidenced."""
    if cfg_name is None:
        return set(query_slots) <= seen
    cfg = L4_CONFIGS[cfg_name]
    defaults = {e for e, _ in cfg["defaults"]}
    pair_of = {}
    for a, b in cfg["pairs"]:
        pair_of[a] = b
        pair_of[b] = a
    for sl in query_slots:
        if sl in defaults:
            continue
        if sl in pair_of:
            if sl not in seen and pair_of[sl] not in seen:
                return False
        elif sl not in seen:
            return False
    return True


def _coverage_parts(states: np.ndarray):
    """History-evidenced slot set and future-queried slot set."""
    T = states.shape[0]
    half = T // 2
    seen = set()
    for t in range(half - 1):
        seen.update(np.unique(slots_of(states[t])).tolist())
    queried = set()
    for t in range(half - 1, T - 1):
        queried.update(np.unique(slots_of(states[t])).tolist())
    return seen, queried


def is_self_consistent(states: np.ndarray, cfg_name: str = None) -> bool:
    """Every slot queried in the future half is predictable from the
    history half (plus family constraints when cfg_name is set)."""
    T = states.shape[0]
    half = T // 2
    seen = set()
    for t in range(half - 1):
        seen.update(np.unique(slots_of(states[t])).tolist())
    for t in range(half - 1, T - 1):
        q = np.unique(slots_of(states[t])).tolist()
        if not _sc_query_check(q, seen, cfg_name):
            return False
    return True


def is_exercised(states: np.ndarray, cfg_name: str) -> bool:
    """Mainline semantics (cawm l4v2_exercised_mask): a default slot is
    future-queried while unevidenced, OR a pair slot is future-queried
    while unevidenced with its partner evidenced (one-hop inference
    forced). Pair slots with NEITHER member evidenced are simply never
    queried under SC, so the family constraint is not exercised there."""
    cfg = L4_CONFIGS[cfg_name]
    seen, queried = _coverage_parts(states)
    for e, _ in cfg["defaults"]:
        if e in queried and e not in seen:
            return True
    for a, b in cfg["pairs"]:
        if b in queried and b not in seen and a in seen:
            return True
        if a in queried and a not in seen and b in seen:
            return True
    return False
