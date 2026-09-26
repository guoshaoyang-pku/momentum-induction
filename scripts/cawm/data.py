"""Data: deterministic on-the-fly trajectory stream (Option A) + pinned eval corpora.

Stream convention (docs/SETTINGS.md §4/§8): sample i of stream s is generated
with rng seed [s, i] — bit-reproducible, order-independent, identical for every
model. Train stream seed 42; eval corpora are materialized, checksummed files
(seeds 43/44) so evaluation never drifts with code changes.

Encoding: storage 0/1; model inputs are ±1 (black = +1) converted at the
boundary; BCE targets are 0/1.
"""

import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import rules as R
from .simulate import neighbor_count, simulate

TRAJ_LEN = 16  # t_in = 8 (frames 0..7), t_out = 8 (frames 8..15)
PREFIX_LEN = 8


def _draw_sample_params(index, master_seed, rule_pool, grid, l4=None, rule_override=None):
    """Per-index deterministic draws (rule, init). Draw order is part of the
    stream convention: rule, [l4 bias], density, init."""
    rng = np.random.default_rng([int(master_seed), int(index)])
    if rule_override is not None:
        rule = int(rule_override)
    else:
        rule = int(rule_pool[rng.integers(rule_pool.size)])
        if l4 is not None:
            rule = R.apply_l4_bias(rule, l4[0], l4[1], l4[2], rng)
    dens = rng.uniform(0.2, 0.8)
    init = (rng.random((grid, grid)) < dens).astype(np.uint8)
    return rule, init


def _simulate_strided(rules, inits, stride):
    """TRAJ_LEN frames sampled every `stride` generations (E11.5, SETTINGS
    S13); stride 1 is the registered protocol, bitwise."""
    if stride == 1:
        return simulate(rules, inits, TRAJ_LEN - 1)
    return simulate(rules, inits, (TRAJ_LEN - 1) * stride)[:, ::stride]


def sample_trajectory(index, master_seed, rule_pool, grid=8, l4=None, rule_override=None,
                      stride=1):
    """Draw trajectory `index` of stream `master_seed`. Returns (rule, frames(16,H,W))."""
    rule, init = _draw_sample_params(index, master_seed, rule_pool, grid, l4, rule_override)
    frames = _simulate_strided(np.array([rule]), init[None], stride)[0]
    return rule, frames


def batch_prefix_coverage(frames):
    """Vectorized prefix coverage for (B,16,H,W) frames -> (B,18) bool."""
    B = frames.shape[0]
    H, W = frames.shape[2], frames.shape[3]
    x = frames[:, :PREFIX_LEN]
    n = neighbor_count(x.reshape(B * PREFIX_LEN, H, W)).reshape(B, PREFIX_LEN, H, W)
    idx = 9 * x.astype(np.int16) + n                          # (B,8,H,W)
    idx7 = idx[:, :PREFIX_LEN - 1]                            # evidence transitions
    flat = (np.arange(B)[:, None, None, None] * R.N_ENTRIES + idx7).ravel()
    cov = np.zeros(B * R.N_ENTRIES, dtype=bool)
    cov[flat] = True
    return cov.reshape(B, R.N_ENTRIES)


def sample_batch(indices, master_seed, rule_pool, grid=8, l4=None, rule_override=None,
                 stride=1):
    """Batch version of sample_trajectory (identical per-index results)."""
    idx = list(indices)
    rules = np.empty(len(idx), dtype=np.int64)
    inits = np.empty((len(idx), grid, grid), dtype=np.uint8)
    for j, i in enumerate(idx):
        rules[j], inits[j] = _draw_sample_params(i, master_seed, rule_pool, grid, l4,
                                                 rule_override)
    frames = _simulate_strided(rules, inits, stride)          # (B,16,H,W)
    covs = batch_prefix_coverage(frames)
    return rules, frames, covs


def prefix_coverage(frames):
    """Slots (s,n) whose outcome is observable within the prefix (single trajectory)."""
    return batch_prefix_coverage(frames[None])[0]


def queried_slots_mask(frames):
    """Vectorized: slots exercised by the FUTURE transitions (source frames
    7..14) -> (B,18) bool. A slot queried but not covered in the prefix is
    not determined by a direct prefix lookup. This mask records missing
    evidence, not a Bayes error rate: a fixed train/test rule split can induce
    dependencies among unobserved bits, and distinct rules can share a future."""
    B = frames.shape[0]
    H, W = frames.shape[2], frames.shape[3]
    x = frames[:, PREFIX_LEN - 1:TRAJ_LEN - 1]                # source frames 7..14
    n = neighbor_count(x.reshape(B * 8, H, W)).reshape(B, 8, H, W)
    idx = 9 * x.astype(np.int16) + n
    flat = (np.arange(B)[:, None, None, None] * R.N_ENTRIES + idx).ravel()
    q = np.zeros(B * R.N_ENTRIES, dtype=bool)
    q[flat] = True
    return q.reshape(B, R.N_ENTRIES)


def unpredictable_cells_mask(frames, covs, prior_slots_mask):
    """Per-cell version of queried_slots_mask -> (B,8,H,W) bool.

    For each output cell (frame k+1, k=0..7; source frame = 7+k), compute its
    slot 9*s+n from the TRUE source frame and its 3x3 neighbour count. The
    cell is UNPREDICTABLE iff its slot is (not covered in covs) AND (not a
    prior slot). prior_slots_mask is a (18,) bool (or None -> all-False).

    On a self-consistent corpus every queried slot is covered (or prior), so
    the mask is all-False for every sample (protocol invariant, test-locked).
    """
    B = frames.shape[0]
    H, W = frames.shape[2], frames.shape[3]
    x = frames[:, PREFIX_LEN - 1:TRAJ_LEN - 1]                # source frames 7..14
    n = neighbor_count(x.reshape(B * 8, H, W)).reshape(B, 8, H, W)
    idx = (9 * x.astype(np.int16) + n).astype(np.int64)       # (B,8,H,W) slot
    covered = np.take_along_axis(covs.reshape(B, 18, 1, 1),
                                 idx, axis=1)                 # (B,8,H,W) bool
    if prior_slots_mask is None:
        prior = np.zeros(18, dtype=bool)
    else:
        prior = np.asarray(prior_slots_mask, dtype=bool)
    is_prior = prior[idx]                                     # (B,8,H,W) bool
    return (~covered) & (~is_prior)


def l4v2_exercised_mask(frames, covs, l4v2):
    """(B,) bool: trajectory actually USES an L4-v2 constraint — a default
    slot queried while uncovered, or a pair slot queried while uncovered
    with its partner covered (one-hop inference forced)."""
    cfg = R.L4V2_CONFIGS[l4v2]
    q = queried_slots_mask(frames)
    ex = np.zeros(frames.shape[0], dtype=bool)
    for e, _ in cfg["defaults"]:
        ex |= q[:, e] & ~covs[:, e]
    for a, b in cfg["pairs"]:
        ex |= q[:, b] & ~covs[:, b] & covs[:, a]
        ex |= q[:, a] & ~covs[:, a] & covs[:, b]
    return ex


def _pool_window(args):
    """Worker for build_exercised_pool: one index window -> accepted
    (rules, frames, covs) that are SC-valid AND exercised."""
    lo, hi, master_seed, l4v2, half, grid = args
    pool = R.l4v2_pool(l4v2, half)
    out_r, out_f, out_c = [], [], []
    for i in range(lo, hi, 4096):
        idx = list(range(i, min(i + 4096, hi)))
        rules, frames, covs = sample_batch(idx, master_seed, pool, grid=grid)
        pred = R.l4v2_effective_coverage(covs, l4v2)
        ok = ~(queried_slots_mask(frames) & ~pred).any(axis=1)
        ok &= l4v2_exercised_mask(frames, covs, l4v2)
        out_r.append(rules[ok]); out_f.append(frames[ok]); out_c.append(covs[ok])
    return (np.concatenate(out_r), np.concatenate(out_f), np.concatenate(out_c))


def build_exercised_pool(path, n, master_seed, l4v2, half="train", grid=8,
                         window=262144, n_procs=32, max_draws=2 * 10 ** 9):
    """E6.6 exercised-oversampling arm: materialize a pinned pool of n
    SC-valid, constraint-EXERCISED training trajectories from the TRAIN half
    of the l4v2 family. Deterministic: index windows [0,window), [window,
    2*window), ... are filtered in order and the first n accepted are kept
    (multiprocessing only parallelizes the windows; order is preserved).
    Stream seed convention: master_seed here must differ from the training
    stream seed so pool and stream never alias the same (seed,index)."""
    import multiprocessing as mp
    rules, frames, covs = [], [], []
    got, drawn, lo = 0, 0, 0
    with mp.Pool(n_procs) as p:
        while got < n and drawn < max_draws:
            jobs = [(lo + k * window, lo + (k + 1) * window, master_seed, l4v2,
                     half, grid) for k in range(n_procs)]
            for r, f, c in p.imap(_pool_window, jobs):
                rules.append(r); frames.append(f); covs.append(c)
                got += len(r)
            lo += n_procs * window
            drawn = lo
    if got < n:
        raise RuntimeError(f"pool {path}: only {got}/{n} exercised trajs in {drawn} draws")
    rules = np.concatenate(rules)[:n]
    frames = np.concatenate(frames)[:n]
    covs = np.concatenate(covs)[:n]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, frames=frames, rules=rules, covs=covs)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {"n": int(n), "master_seed": int(master_seed), "half": half,
            "grid": int(grid), "l4v2": l4v2, "draws": int(drawn),
            "self_consistent": True, "require_exercised": True, "sha256": sha,
            "protocol": "post-submission regenerated (seed=42 split), hash-split protocol"}
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


class StreamDataset(Dataset):
    """Deterministic infinite-i.i.d. trajectory stream; length = planned samples."""

    def __init__(self, master_seed, half="train", grid=8, length=10 ** 6, l4=None,
                 rule_override=None, stride=1):
        assert half in ("train", "zs")
        self.stride = int(stride)
        self.master_seed = master_seed
        self.grid = grid
        self.length = length
        self.l4 = l4
        self.rule_override = rule_override
        self.rule_pool = R.rule_split()[0] if half == "train" else R.rule_split()[1]

    def __len__(self):
        return self.length

    def get_batch(self, indices):
        """Vectorized batch draw. Returns dict of torch tensors."""
        rules, frames, covs = sample_batch(indices, self.master_seed, self.rule_pool,
                                           grid=self.grid, l4=self.l4,
                                           rule_override=self.rule_override,
                                           stride=self.stride)
        return {
            "frames": torch.from_numpy(frames),
            "rule": torch.from_numpy(rules),
            "cov": torch.from_numpy(covs),
        }

    def __getitem__(self, i):
        rule, frames = sample_trajectory(i, self.master_seed, self.rule_pool,
                                          grid=self.grid, l4=self.l4,
                                          rule_override=self.rule_override,
                                          stride=self.stride)
        cov = prefix_coverage(frames)
        return {
            "frames": torch.from_numpy(frames.astype(np.uint8)),  # (16,H,W) 0/1
            "rule": torch.tensor(rule, dtype=torch.long),
            "cov": torch.from_numpy(cov),
        }


class ChunkedStreamDataset(Dataset):
    """One item = one whole training batch (chunk_size contiguous stream indices).

    Chunk c covers stream indices [c*chunk_size, (c+1)*chunk_size) and is built
    with the same vectorized `sample_batch` call as the legacy synchronous path,
    so DataLoader workers (any count) produce bit-identical batches. Feeding
    this to torch.utils.data.DataLoader(batch_size=None, num_workers=W) overlaps
    trajectory simulation with model forward/backward.

    sc_filter=True (E3.5, protocol note 2026-08-30) rejection-samples the
    TRAINING stream to the SC protocol — the same filter as the `_sc` eval
    corpora: keep a trajectory only if every queried slot is covered by prefix
    evidence or belongs to prior_slots. Deterministic: chunk c draws the raw
    index window [c*chunk_size*sc_oversample, (c+1)*chunk_size*sc_oversample)
    and keeps the first chunk_size accepted trajectories, so batches depend
    only on (master_seed, chunk index), never on worker scheduling. With SC
    yield ~0.65 and sc_oversample=4 the acceptance shortfall probability is
    negligible; a shortfall raises rather than silently reusing indices.
    Default OFF = bit-identical legacy behaviour."""

    def __init__(self, master_seed, half="train", grid=8, n_chunks=1000,
                 chunk_size=128, l4=None, rule_override=None, start_chunk=0,
                 sc_filter=False, sc_oversample=4, prior_slots=(), l4v2=None,
                 ex_pool=None, ex_frac=0.0, stride=1):
        assert half in ("train", "zs")
        self.stride = int(stride)
        # E6.6 exercised oversampling (default OFF = bit-identical): replace
        # the first k = round(ex_frac*chunk_size) trajectories of every chunk
        # with entries of a pinned exercised pool, addressed deterministically
        # by (chunk, j) -> pool[(chunk*k + j) % pool_n].
        self.ex_k = int(round(ex_frac * chunk_size)) if ex_pool else 0
        if self.ex_k > 0:
            assert sc_filter and l4v2 is not None, "ex_pool needs sc_filter + l4v2"
            with np.load(ex_pool) as z:
                self.ex_frames = z["frames"]; self.ex_rules = z["rules"]
                self.ex_covs = z["covs"]
            self.ex_n = self.ex_frames.shape[0]
        self.master_seed = master_seed
        self.grid = grid
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.l4 = l4
        self.rule_override = rule_override
        self.start_chunk = int(start_chunk)
        self.sc_filter = bool(sc_filter)
        self.sc_oversample = int(sc_oversample)
        self.l4v2 = l4v2
        self.prior_m = np.zeros(R.N_ENTRIES, dtype=bool)
        for s in prior_slots:
            self.prior_m[s] = True
        if l4v2 is not None:
            self.rule_pool = R.l4v2_pool(l4v2, half)
        else:
            self.rule_pool = R.rule_split()[0] if half == "train" else R.rule_split()[1]

    def __len__(self):
        return self.n_chunks - self.start_chunk

    def __getitem__(self, c):
        chunk = self.start_chunk + int(c)
        if self.sc_filter:
            raw = self.chunk_size * self.sc_oversample
            idx = list(range(chunk * raw, (chunk + 1) * raw))
            rules, frames, covs = sample_batch(idx, self.master_seed, self.rule_pool,
                                               grid=self.grid, l4=self.l4,
                                               rule_override=self.rule_override,
                                               stride=self.stride)
            if self.l4v2 is not None:
                pred = R.l4v2_effective_coverage(covs, self.l4v2)
            else:
                pred = covs | self.prior_m
            bad = queried_slots_mask(frames) & ~pred
            keep = np.flatnonzero(~bad.any(axis=1))
            if keep.size < self.chunk_size:
                raise RuntimeError(
                    f"sc_filter chunk {chunk}: {keep.size}/{self.chunk_size} "
                    f"accepted at oversample {self.sc_oversample}")
            keep = keep[:self.chunk_size]
            rules, frames, covs = rules[keep], frames[keep], covs[keep]
            if self.ex_k > 0:
                j = (chunk * self.ex_k + np.arange(self.ex_k)) % self.ex_n
                rules = np.concatenate([self.ex_rules[j], rules[self.ex_k:]])
                frames = np.concatenate([self.ex_frames[j], frames[self.ex_k:]])
                covs = np.concatenate([self.ex_covs[j], covs[self.ex_k:]])
        else:
            base = chunk * self.chunk_size
            idx = list(range(base, base + self.chunk_size))
            rules, frames, covs = sample_batch(idx, self.master_seed, self.rule_pool,
                                               grid=self.grid, l4=self.l4,
                                               rule_override=self.rule_override,
                                               stride=self.stride)
        return {
            "frames": torch.from_numpy(frames),
            "rule": torch.from_numpy(rules),
            "cov": torch.from_numpy(covs),
        }


def collate(batch):
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "rule": torch.stack([b["rule"] for b in batch]),
        "cov": torch.stack([b["cov"] for b in batch]),
    }


def build_eval_corpus(path, n, master_seed, half="zs", grid=8, l4=None,
                      rule_override=None, full_coverage_only=False, max_draws=None,
                      self_consistent=False, prior_slots=(), l4v2=None,
                      require_exercised=False, stride=1):
    """Materialize a pinned eval corpus as .npz + .json (protocol tags + sha256).

    self_consistent=True turns on rejection sampling (protocol note
    2026-08-30): keep a trajectory only if every slot QUERIED by its realized future
    transitions is covered by the prefix evidence (or belongs to
    prior_slots — L4's slot 9, whose value the rule family fixes by
    construction via apply_l4_bias, so it stays Bayes-predictable when
    uncovered). The filter is evaluated during corpus construction from the
    complete simulated trajectory, before model evaluation. It creates a
    conditional benchmark of answerable worlds; the future is never supplied
    as model input. Without it ~35% of trajectories query rule entries absent
    from the prefix. This is not an information-theoretic SeqAcc bound.

    l4v2 (E6.2/E6.3): draw rules from the constrained family pool
    (R.l4v2_pool — split decided on the constrained rule, no preimage leak)
    and judge predictability with R.l4v2_effective_coverage (defaults always
    predictable; a negation-pair slot predictable iff either member covered).

    require_exercised=True (l4v2 only): additionally keep only trajectories
    where some constraint is actually USED — a default slot queried while
    uncovered, or a pair slot queried while uncovered with its partner
    covered. Standard SC corpora exercise the constraints rarely (l4b: 2/2048
    trajs), so per-stratum numbers there are anecdotes; these corpora make
    the exercised stratum the whole corpus."""
    if l4v2 is not None:
        pool = R.l4v2_pool(l4v2, half)
    else:
        pool = R.rule_split()[0] if half == "train" else R.rule_split()[1]
    assert not (require_exercised and l4v2 is None), \
        "require_exercised only defined for l4v2 corpora"
    prior_m = np.zeros(R.N_ENTRIES, dtype=bool)
    for s in prior_slots:
        prior_m[s] = True
    rules, frames, covs = [], [], []
    i = 0
    cap = max_draws if max_draws is not None else max(50 * n, 50000)
    while len(rules) < n and i < cap:
        take = min(256, cap - i)
        idx = list(range(i, i + take))
        i += take
        b_rules, b_frames, b_covs = sample_batch(idx, master_seed, pool, grid=grid,
                                                 l4=l4, rule_override=rule_override,
                                                 stride=stride)
        keep = np.ones(take, dtype=bool)
        if full_coverage_only:
            keep &= b_covs.all(axis=1)
        if self_consistent:
            if l4v2 is not None:
                pred = R.l4v2_effective_coverage(b_covs, l4v2)
            else:
                pred = b_covs | prior_m
            bad = queried_slots_mask(b_frames) & ~pred
            keep &= ~bad.any(axis=1)
        if require_exercised:
            keep &= l4v2_exercised_mask(b_frames, b_covs, l4v2)
        rules.extend(b_rules[keep].tolist())
        frames.extend(b_frames[keep])
        covs.extend(b_covs[keep])
    if len(rules) < n:
        raise RuntimeError(f"corpus {path}: only {len(rules)}/{n} collected in {cap} draws")
    rules, frames, covs = rules[:n], frames[:n], covs[:n]   # trim final-chunk surplus
    frames = np.stack(frames)
    rules = np.array(rules, dtype=np.int64)
    covs = np.stack(covs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, frames=frames, rules=rules, covs=covs)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {
        "n": int(len(rules)), "master_seed": int(master_seed), "half": half,
        "grid": int(grid), "l4": list(l4) if l4 else None,
        "rule_override": int(rule_override) if rule_override is not None else None,
        "full_coverage_only": bool(full_coverage_only),
        "self_consistent": bool(self_consistent),
        "prior_slots": list(prior_slots),
        "l4v2": l4v2,
        "require_exercised": bool(require_exercised),
        "draws": int(i),
        "sha256": sha,
        "protocol": "post-submission regenerated (seed=42 split), hash-split protocol",
    }
    if stride != 1:
        meta["frame_stride"] = int(stride)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_eval_corpus(path):
    with np.load(path) as z:
        frames = z["frames"]
        rules = z["rules"]
        covs = z["covs"]
    with open(path + ".json") as f:
        meta = json.load(f)
    return {"frames": frames, "rules": rules, "covs": covs, "meta": meta}
