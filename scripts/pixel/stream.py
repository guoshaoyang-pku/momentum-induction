"""Infinite training stream + fixed eval sets for the pixel lab.

One batch shares one hold value k (so tensors collate); scale/offset vary
per trajectory. Cell-space simulation and the self-consistency rejection
happen before rendering, so the protocol is identical to the mainline's
rejection-sampled full-context data, just rendered.

Levels:
  L1   one fixed rule (default B2/S3), no rejection needed.
  L23  random train-half rules, rejection until every future slot is
       evidenced in history.
  L4/L4A/L4B  rules from the constrained family (mainline configs: slot-9
       default / four marginal defaults / negation pairs), hash-split on
       the CONSTRAINED rule id so train and ZS families are disjoint;
       rejection with family-aware predictability (defaults always
       predictable, pair slots predictable from either member). Eval
       additionally requires the trajectory to EXERCISE the constraint
       (query a default without evidence / query the unevidenced pair
       member), matching the mainline's exercised corpora.
"""
import numpy as np
import torch
from torch.utils.data import IterableDataset

from . import ca as ca_mod
from . import render as render_mod

L4_LEVELS = {"L4": "l4", "L4A": "l4a", "L4B": "l4b"}


def _l4_cfg(level):
    return L4_LEVELS.get(level)


class PixelStream(IterableDataset):
    def __init__(self, level="L1", s=8, frame=128, offsets=((32, 32),),
                 k_max=1, batch=32, seed=0, p_init=0.3, l1_rule=ca_mod.RULE_B2S3,
                 hist_states=8, fut_states=8, border=1, rule_pool=0,
                 p_init_hi=0.0, ex_train=0.0, ex_pool=""):
        super().__init__()
        assert level in ("L1", "L23", "L4", "L4A", "L4B")
        self.level = level
        self.cfg = _l4_cfg(level)
        self.ex_train = ex_train  # fraction of batch slots forced exercised
        self.ex_pool = None
        if ex_pool:
            d = np.load(ex_pool)
            self.ex_pool = d["states"]  # (M,16,8,8) uint8, SC+exercised
        self.s = s
        self.frame = frame
        self.offsets = list(offsets)
        self.k_max = k_max
        self.batch = batch
        self.seed = seed
        self.p_init = p_init
        self.p_init_hi = p_init_hi  # if > 0, sample p per traj ~ U(p_init, hi)
        self.l1_rule = l1_rule
        self.hist_states = hist_states
        self.fut_states = fut_states
        self.border = border
        self.pool = None
        if rule_pool > 0 and level != "L1":
            # fixed pool of train-half rules, identical across workers and
            # runs (own seed), so trajectories repeat rules
            prng = np.random.default_rng(777)
            pool = set()
            while len(pool) < rule_pool:
                r = int(prng.integers(0, 1 << ca_mod.N_SLOTS))
                if _rule_ok(r, self.cfg, train=True):
                    pool.add(r)
            self.pool = sorted(pool)

    def _sample_rule(self, rng):
        if self.level == "L1":
            return self.l1_rule
        if self.pool is not None:
            return self.pool[rng.integers(0, len(self.pool))]
        while True:
            r = int(rng.integers(0, 1 << ca_mod.N_SLOTS))
            if _rule_ok(r, self.cfg, train=True):
                return r

    def _sample_traj(self, rng, exercised=False):
        table = ca_mod.table_from_rule(self._sample_rule(rng))
        steps = self.hist_states + self.fut_states
        p = (float(rng.uniform(self.p_init, self.p_init_hi))
             if self.p_init_hi > 0 else self.p_init)
        if self.level == "L1":
            return ca_mod.trajectory(table, rng, steps, p)
        for _ in range(500 if not exercised else 2000):
            st = ca_mod.trajectory(table, rng, steps, p)
            if not ca_mod.is_self_consistent(st, self.cfg):
                continue
            if exercised and not ca_mod.is_exercised(st, self.cfg):
                continue
            return st
        if exercised:
            return None  # caller retries with a fresh rule
        # degenerate rule (grid freezes): resample the rule instead
        return self._sample_traj(rng)

    def _make_batch(self, rng):
        k = int(rng.integers(1, self.k_max + 1))
        B = self.batch
        T = self.hist_states + self.fut_states
        hists = np.empty((B, self.hist_states * k, 1, self.frame, self.frame), np.float32)
        futs = np.empty((B, self.fut_states * k, 1, self.frame, self.frame), np.float32)
        states = np.empty((B, T, ca_mod.GRID, ca_mod.GRID), np.uint8)
        geoms = np.empty((B, 2), np.int64)
        n_ex = int(round(B * self.ex_train)) if self.cfg else 0
        for b in range(B):
            if b < n_ex:
                if self.ex_pool is not None:
                    st = self.ex_pool[rng.integers(0, len(self.ex_pool))]
                else:
                    st = None
                    while st is None:
                        st = self._sample_traj(rng, exercised=True)
            else:
                st = self._sample_traj(rng)
            off = self.offsets[rng.integers(0, len(self.offsets))]
            vid = render_mod.render_states(st, self.s, off, self.frame, k,
                                           border=self.border)
            cut = self.hist_states * k
            hists[b], futs[b] = vid[:cut], vid[cut:]
            states[b] = st
            geoms[b] = off
        return {"hist": torch.from_numpy(hists), "fut": torch.from_numpy(futs),
                "states": torch.from_numpy(states).long(),
                "geom": torch.from_numpy(geoms), "k": k}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = self.seed + (info.id if info is not None else 0)
        rng = np.random.default_rng(seed)
        while True:
            yield self._make_batch(rng)


def _rule_ok(rule, cfg, train):
    half_ok = ca_mod.is_train_rule(rule) if train \
        else not ca_mod.is_train_rule(rule)
    if not half_ok:
        return False
    if cfg is not None and not ca_mod.rule_satisfies(rule, cfg):
        return False
    return True


def make_eval_batches(level="L1", s=8, frame=128, offsets=((32, 32),),
                      k_values=(1,), n_traj=64, seed=999, p_init=0.3,
                      l1_rule=ca_mod.RULE_B2S3, hist_states=8, fut_states=8,
                      border=1, oversample=64, exercised=True):
    """Fixed eval batches (one per k). Non-L1 levels draw rules from the
    held-out half (and the family, for L4 levels). With exercised=True,
    L4 eval trajectories must also exercise the family constraint; those
    are rare (mainline: ~0.02% for l4b), so we oversample trajectories
    per rule and keep the exercised ones."""
    cfg = _l4_cfg(level)
    out = []
    for k in k_values:
        rng = np.random.default_rng(seed + k)
        T = hist_states + fut_states
        hists = np.empty((n_traj, hist_states * k, 1, frame, frame), np.float32)
        futs = np.empty((n_traj, fut_states * k, 1, frame, frame), np.float32)
        states = np.empty((n_traj, T, ca_mod.GRID, ca_mod.GRID), np.uint8)
        geoms = np.empty((n_traj, 2), np.int64)
        b = 0
        while b < n_traj:
            if level == "L1":
                st = ca_mod.trajectory(ca_mod.table_from_rule(l1_rule), rng,
                                       T, p_init)
                cands = [st]
            else:
                while True:
                    rule = int(rng.integers(0, 1 << ca_mod.N_SLOTS))
                    if _rule_ok(rule, cfg, train=False):
                        break
                table = ca_mod.table_from_rule(rule)
                want_ex = cfg is not None and exercised
                n_scan = oversample * (n_traj - b) if want_ex else 500
                cands = []
                for _ in range(n_scan):
                    cand = ca_mod.trajectory(table, rng, T, p_init)
                    if not ca_mod.is_self_consistent(cand, cfg):
                        continue
                    if want_ex and not ca_mod.is_exercised(cand, cfg):
                        continue
                    cands.append(cand)
                    if len(cands) >= n_traj - b:
                        break
                if not cands:
                    continue  # next rule
            for st in cands:
                if b >= n_traj:
                    break
                off = offsets[rng.integers(0, len(offsets))]
                vid = render_mod.render_states(st, s, off, frame, k,
                                               border=border)
                cut = hist_states * k
                hists[b], futs[b] = vid[:cut], vid[cut:]
                states[b] = st
                geoms[b] = off
                b += 1
        out.append({"hist": torch.from_numpy(hists), "fut": torch.from_numpy(futs),
                    "states": torch.from_numpy(states).long(),
                    "geom": torch.from_numpy(geoms), "k": k})
    return out
