"""Outer-totalistic CA rule space and the train/ZS hash split.

Convention (frozen in docs/SETTINGS.md §2): rule index is an 18-bit table;
bit i is the output for entry i, with entry index = 9*s + n (s = own state,
n = live-neighbor count).
"""

import numpy as np

RULE_SPACE_SIZE = 2 ** 18  # 262,144
N_ENTRIES = 18

# Game of Life as an outer-totalistic rule: survival (s=1) at n in {2,3},
# birth (s=0) at n=3.
GOL_RULE = (1 << (9 * 1 + 2)) | (1 << (9 * 1 + 3)) | (1 << (9 * 0 + 3))

_SPLIT_CACHE: dict = {}


def rule_split(seed: int = 42):
    """Deterministic unbiased train/ZS split of the full rule space.

    Rule-index bits ARE table entries, so parity/low-bit-modulo splits are
    systematically biased on a single entry; a fixed-seed permutation is not.
    Returns (train_rules, zs_rules), each sorted, disjoint, union = full space.
    """
    if seed not in _SPLIT_CACHE:
        perm = np.random.default_rng(seed).permutation(RULE_SPACE_SIZE)
        half = RULE_SPACE_SIZE // 2
        _SPLIT_CACHE[seed] = (np.sort(perm[:half]), np.sort(perm[half:]))
    return _SPLIT_CACHE[seed]


def apply_l4_bias(rule: int, entry: int, default: int, prior: float, rng) -> int:
    """L4 rule-distribution bias: with prob `prior`, force entry `entry` to `default`."""
    if rng.random() < prior:
        return (rule & ~(1 << entry)) | (int(default) << entry)
    return rule


# ------------------------------------------------------------------ L4-v2
# (EXPERIMENT_PLAN E6.2/E6.3, protocol note 2026-09-02: the legacy
# single-slot L4 is a degenerate special case).

L4V2_CONFIGS = {
    # E6.2 L4A: multi-slot marginal defaults — family-fixed, always binding.
    # Mixed own-state rows and mixed colours (two black, two white) to kill
    # the colour-asymmetry confound of the legacy single-slot L4.
    "l4a": {"defaults": ((2, 1), (9, 1), (6, 0), (16, 0)), "pairs": ()},
    # E6.3 L4B: relational prior — negation pairs b = NOT a with b = a + 9
    # (same neighbour count, opposite own state): evidence for either member
    # determines the other by the family constraint (one-hop inference).
    # Slots disjoint from the L4A defaults.
    "l4b": {"defaults": (), "pairs": ((4, 13), (3, 12))},
}

_L4V2_POOL_CACHE: dict = {}


def l4v2_pool(cfg_name: str, half: str, seed: int = 42):
    """All rules satisfying the family constraints, intersected with the
    standard hash-split half. Membership is decided on the CONSTRAINED rule
    id itself, so the train and ZS families are truly disjoint — forcing
    bits AFTER a raw-rule draw (the legacy apply_l4_bias path) lets the two
    halves collide on the constrained rule (preimage leak); this does not."""
    key = (cfg_name, half, seed)
    if key not in _L4V2_POOL_CACHE:
        cfg = L4V2_CONFIGS[cfg_name]
        ids = np.arange(RULE_SPACE_SIZE, dtype=np.int64)
        keep = np.ones(RULE_SPACE_SIZE, dtype=bool)
        for e, v in cfg["defaults"]:
            keep &= ((ids >> e) & 1) == v
        for a, b in cfg["pairs"]:
            keep &= ((ids >> a) & 1) != ((ids >> b) & 1)
        fam = ids[keep]
        tr, zs = rule_split(seed)
        _L4V2_POOL_CACHE[key] = np.intersect1d(
            fam, tr if half == "train" else zs)
    return _L4V2_POOL_CACHE[key]


def l4v2_effective_coverage(covs: np.ndarray, cfg_name: str) -> np.ndarray:
    """(B,18) evidence coverage -> (B,18) PREDICTABILITY under the family
    constraints: a default slot is always predictable; a pair slot is
    predictable iff either member is covered (negation is symmetric)."""
    cfg = L4V2_CONFIGS[cfg_name]
    eff = covs.copy()
    for e, _ in cfg["defaults"]:
        eff[:, e] = True
    for a, b in cfg["pairs"]:
        both = covs[:, a] | covs[:, b]
        eff[:, a] = both
        eff[:, b] = both
    return eff
