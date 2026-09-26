'''Tests for the eval-only commit-schedule extension (scripts/cawm/commit_schedule.py).'''

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cawm.commit_schedule import commit_rollout  # noqa: E402
from cawm.models.diffusion import T_STEPS  # noqa: E402
from cawm.models.diffusion_vanilla import DiffusionVanilla  # noqa: E402

HI = 8.0  # |tanh(HI / 2)| ~ 0.99933 -> passes the 0.99 gate
LO = 0.1  # |tanh(LO / 2)| ~ 0.050   -> fails the gate


def make_prefix(batch, height, width, seed=0):
    gen = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (batch, 1, 8, height, width), generator=gen)
    return bits.float() * 2.0 - 1.0


class ChainToy:
    '''One-step transition denoiser.

    Frame 0 is read off the last history frame.  Frame i > 0 is the sign-flip
    of frame i-1, but ONLY when frame i-1 is already committed (level 0);
    otherwise its logits stay low-confidence.  Already-committed frames get
    adversarial sign-flipped logits: if the scheduler ever re-used them, the
    final predictions would break the expected chain.
    '''

    t_per_frame = True

    def __init__(self):
        self.calls = 0
        self.records = []

    def denoise(self, x, hist, levels):
        self.calls += 1
        self.records.append((x.clone(), hist.clone(), levels.clone()))
        logits = torch.full_like(x, LO)
        logits[:, 0] = HI * hist[:, -1]
        for i in range(1, x.shape[1]):
            prev_clean = levels[:, i - 1] == 0
            if bool(prev_clean.any()):
                logits[prev_clean, i] = -HI * x[prev_clean, i - 1]
        committed = (levels == 0)[:, :, None, None]
        logits = torch.where(committed, -HI * x, logits)
        return logits, logits


class SpeedToy:
    '''Per-world confidence speed, encoded in the history so it survives
    incomplete-subset batching: speed = number of +1 cells in history frame 0.
    Each call unlocks high confidence for pending frames whose index is below
    committed_count + speed.'''

    t_per_frame = True

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def denoise(self, x, hist, levels):
        self.calls += 1
        self.batch_sizes.append(int(x.shape[0]))
        speed = (hist[:, 0] > 0).flatten(1).sum(dim=1)
        done = (levels == 0).sum(dim=1)
        frame = torch.arange(x.shape[1])
        unlocked = frame[None, :] < (done + speed)[:, None]
        hi = (unlocked & (levels != 0))[:, :, None, None]
        logits = torch.where(hi, torch.full_like(x, HI), torch.full_like(x, LO))
        return logits, logits


class ConstToy:
    '''Uniform high-confidence positive logits everywhere.'''

    t_per_frame = True

    def __init__(self):
        self.calls = 0

    def denoise(self, x, hist, levels):
        self.calls += 1
        logits = torch.full_like(x, HI)
        return logits, logits


def expected_chain_u8(prefix):
    last = prefix[:, 0, -1]
    frames = [last * ((-1.0) ** i) for i in range(8)]
    return (torch.stack(frames, dim=1) > 0).to(torch.uint8)


def speed_prefix(speeds, height=3, width=3):
    prefix = -torch.ones(len(speeds), 1, 8, height, width)
    for b, s in enumerate(speeds):
        prefix[b, 0, 0].view(-1)[:s] = 1.0
    return prefix


def test_causal_chain_predictions_and_counters():
    torch.manual_seed(0)
    prefix = make_prefix(2, 3, 3)
    model = ChainToy()
    preds, rounds, per_world, calls = commit_rollout(model, prefix, 'causal')
    assert preds.dtype == torch.uint8 and tuple(preds.shape) == (2, 8, 3, 3)
    assert rounds.dtype == torch.int64 and per_world.dtype == torch.int64
    # commits are hard-frozen: the adversarial flipped logits emitted for
    # committed frames never leak into the final predictions
    assert torch.equal(preds, expected_chain_u8(prefix))
    assert torch.equal(rounds, torch.arange(8).expand(2, 8))
    assert torch.equal(per_world, torch.full((2,), 8, dtype=torch.int64))
    assert calls == 8 and model.calls == 8


def test_confident_needs_committed_predecessor():
    torch.manual_seed(0)
    prefix = make_prefix(2, 3, 3)
    preds, rounds, _, calls = commit_rollout(ChainToy(), prefix, 'confident')
    # frame i only becomes confident after frame i-1 is committed, so the
    # confidence-greedy order collapses to strictly causal order
    assert torch.equal(rounds, torch.arange(8).expand(2, 8))
    assert torch.equal(preds, expected_chain_u8(prefix))
    assert calls == 8


def test_parallel_no_within_round_cascade():
    torch.manual_seed(0)
    prefix = make_prefix(1, 3, 3)
    preds, rounds, _, calls = commit_rollout(ChainToy(), prefix, 'parallel', threshold=0.99)
    # selections are fixed before any commit of the round, so frame i+1 can
    # never ride on frame i being committed in the same round
    assert torch.equal(rounds, torch.arange(8).expand(1, 8))
    assert calls == 8
    assert torch.equal(preds, expected_chain_u8(prefix))


def test_pending_canvas_held_and_levels():
    prefix = make_prefix(1, 3, 3)
    gen = torch.Generator().manual_seed(5)
    noise = torch.randn(1, 8, 3, 3, generator=gen)
    model = ChainToy()
    preds, _, _, calls = commit_rollout(model, prefix, 'causal', initial_noise=noise)
    assert calls == 8 and len(model.records) == 8
    for r, (x, hist, levels) in enumerate(model.records):
        assert torch.equal(hist, prefix[:, 0])
        assert torch.equal(levels[0, :r], torch.zeros(r, dtype=torch.int64))
        assert torch.equal(levels[0, r:], torch.full((8 - r,), T_STEPS - 1, dtype=torch.int64))
        # pending frames hold the ORIGINAL noise canvas, unchanged every round
        assert torch.equal(x[0, r:], noise[0, r:])
        # committed frames are hard +-1 spins of the returned predictions
        assert torch.equal(x[0, :r], preds[0, :r].float() * 2.0 - 1.0)


def test_parallel_speeds_batching_and_fallback():
    speeds = [8, 4, 1, 0]
    prefix = speed_prefix(speeds)
    model = SpeedToy()
    torch.manual_seed(0)
    preds, rounds, per_world, calls = commit_rollout(model, prefix, 'parallel', threshold=0.99)
    # worlds leave the batch as soon as they complete
    assert model.batch_sizes == [4, 3, 2, 2, 2, 2, 2, 2]
    assert calls == 8 and model.calls == 8
    assert torch.equal(per_world, torch.tensor([1, 2, 8, 8], dtype=torch.int64))
    assert torch.equal(rounds[0], torch.zeros(8, dtype=torch.int64))
    assert torch.equal(rounds[1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert torch.equal(rounds[2], torch.arange(8))
    # the speed-0 world never crosses the gate: the fallback commits the
    # single highest-confidence pending frame (exact tie -> earliest), one
    # per round, guaranteeing progress
    assert torch.equal(rounds[3], torch.arange(8))
    # complete partition: every frame committed exactly once
    assert bool((rounds >= 0).all())
    assert bool((preds == 1).all())  # all toy logits are positive


def test_confident_exact_tie_is_deterministic_earliest():
    torch.manual_seed(0)
    prefix = make_prefix(2, 3, 3)
    _, rounds, _, calls = commit_rollout(ConstToy(), prefix, 'confident')
    assert torch.equal(rounds, torch.arange(8).expand(2, 8))
    assert calls == 8


def test_random_reproducible_with_fixed_generator():
    prefix = make_prefix(3, 3, 3)

    def run(order_seed):
        torch.manual_seed(0)  # pins the default initial noise draw
        return commit_rollout(ConstToy(), prefix, 'random', order_seed=order_seed)

    preds_a, rounds_a, per_world_a, calls_a = run(7)
    preds_b, rounds_b, per_world_b, calls_b = run(7)
    assert torch.equal(preds_a, preds_b)
    assert torch.equal(rounds_a, rounds_b)
    assert torch.equal(per_world_a, per_world_b)
    assert calls_a == calls_b == 8
    # one commit per world per round, in a generator-chosen order
    sorted_rounds, _ = torch.sort(rounds_a, dim=1)
    assert torch.equal(sorted_rounds, torch.arange(8).expand(3, 8))


def test_causal_parity_with_vanilla_frame_ar_rollout():
    torch.manual_seed(0)
    model = DiffusionVanilla(grid=4, t_per_frame=True, n_layers=4)
    model.eval()
    prefix = make_prefix(2, 4, 4, seed=1)

    torch.manual_seed(1234)
    base = model.rollout(prefix, nfe=1, schedule='frame_ar', commit=True, pending='noise')
    if isinstance(base, (tuple, list)):
        base = base[0]

    torch.manual_seed(1234)
    preds, rounds, per_world, calls = commit_rollout(model, prefix, 'causal')
    assert calls == 8
    assert torch.equal(rounds, torch.arange(8).expand(2, 8))
    assert torch.equal(per_world, torch.full((2,), 8, dtype=torch.int64))
    base_u8 = (base.to(torch.float32) > 0).to(torch.uint8)
    assert tuple(base_u8.shape) == tuple(preds.shape)
    assert torch.equal(base_u8, preds)


def test_validation_errors():
    prefix = make_prefix(1, 3, 3)
    model = ConstToy()
    with pytest.raises(ValueError):
        commit_rollout(model, prefix, 'bogus')
    with pytest.raises(ValueError):
        commit_rollout(model, prefix[:, 0], 'causal')  # wrong rank
    with pytest.raises(ValueError):
        commit_rollout(model, prefix * 0.5, 'causal')  # not +-1
    with pytest.raises(ValueError):
        commit_rollout(model, prefix, 'parallel', threshold=0.0)
    with pytest.raises(ValueError):
        commit_rollout(model, prefix, 'parallel', threshold=1.5)
    with pytest.raises(ValueError):
        commit_rollout(model, prefix, 'causal', initial_noise=torch.zeros(1, 8, 2, 2))

    class NotPerFrame:
        t_per_frame = False

        def denoise(self, x, hist, levels):
            raise AssertionError('should not be called')

    with pytest.raises(ValueError):
        commit_rollout(NotPerFrame(), prefix, 'causal')
