# Exact trajectory MAP: directional learned CA

This is the independent 50-slot directional-rule experiment, not original R0
or a Structance measurement. It uses the unchanged learned rank-two metric
(20 trainable parameters in the earlier model) from the M=2048 runs in
PR #13, with no additional training. Its learned partition agrees with
the 50 ground-truth classes for all 512 neighborhoods and all three seeds.
The decoder gets only the first eight frames and the learned partition,
not the true classes or future labels. Unknown rule outputs have independent
fair Bernoulli priors and known rule outputs are fixed by prefix evidence.

## Exact search

For a possible future trajectory, every newly queried previously unknown
rule output has to take one specific value. Thus its marginal probability,
summing over all unqueried bits, is exactly 2^(-u), with u the number of
distinct newly queried slots. The cost never decreases as the trajectory
extends. Branch only on rule slots needed by the next state; assignments
are persistent throughout the rollout. Prune when the partial cost cannot
beat the incumbent. Certify when all improving branches are exhausted or the
first-step lower bound is attained. A node cap never implies certification.
Ties retain the incumbent; optimal trajectories need not be unique.

Use the exact same 32-candidate subset of the 128-candidate RNG pool in
PR #14, not a fresh 32-candidate draw. The runner verifies all six prior
baseline metrics and data hashes exactly. Three seeds, 2048 trajectories
per seed on each of the original and fresh validation splits; 12,288 total.

| Split | 32 SeqAcc8 | Exact SeqAcc8 | 32 predicted mass | Exact predicted mass | Certified | Mean visited nodes |
|---|---:|---:|---:|---:|---:|---:|
| Original | 33.7565% | 33.7565% | 33.5279% | 33.5462% | 100% | 9.18 |
| Fresh | 34.1471% | 34.1309% | 33.8678% | 33.8839% | 100% | 8.79 |

All 12,288 exact searches certified within the 4096-node cap (largest
search 656 nodes). The optimal probability increases on about 1.1% of
worlds but its mean improvement is only 0.016–0.018 percentage points.
Nonpersistent-world SeqAcc8 is 32.42% on original and 32.85% on fresh,
essentially unchanged. Exact MAP improves expected accuracy under the
model prior, not necessarily realized accuracy on a fixed finite sample.
The nonpersistent cohort is defined using actual futures for evaluation
only: its conditional accuracy is not directly a posterior probability.
The learned geometry already recovers the class partition. Under this
independent fair-rule prior, more search cannot substantially improve the
eight-step accuracy; hidden rule bits remain unobserved. This is not a
universal upper bound for other distributions or priors.

Reproduction (use new output paths if results already exist):
```sh
PYTHONPATH=scripts python -m pytest -q scripts/tests/test_directional_exact.py scripts/tests/test_directional_search.py
python scripts/run_directional_exact.py --out /tmp/directional-exact-fresh.jsonl
python scripts/run_directional_exact.py --data-seed-base 30000 --out /tmp/directional-exact-original.jsonl
python scripts/plot_directional_exact.py
```

Tests: 48 independently enumerated tiny-rule problems, capped search
certification, prefix-only invariance, relabeling invariance and prior
search tests. No new model training, GPU use, checkpoint commit or
modification to the R0 data/code.
