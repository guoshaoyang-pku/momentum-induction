# Persistent rule search on the learned directional CA

## Motivation and setting

The previous 50-slot directional-rule experiment (d1f6267, PR #13)
learned the local geometry but greedy rollouts remained limited by unseen
rule bits. Here we test whether persistent rule hypotheses and trajectory-level
probability improve closed-loop prediction without new training or true future
inputs. This is a synthetic non-R0 control, not a Structance measurement.

Use the three existing M=2048 rank-2 checkpoints (20 learned parameters,
800 Adam updates) stored in the small coefficient records in
results/directional_metric/m2048.jsonl. The fixed .5 affinity threshold
(also used by the earlier rule_hypotheses experiment, 0b41701) forms
connected components over all 512 raw binary neighborhoods. No true slot
labels are used by the learned decoder. The diagnostic comparison against
the known 50 groups happens outside decoding. The partitions match exactly
for all three seeds, and canonical relabeling makes the two decoders produce
identical outputs under common random draws.

The rule table is inferred only from transitions in frames 0..7.
Known bits are fixed, unknown bits have independent fair Bernoulli priors.
For each hypothesis we hold a sampled complete table fixed for all eight
steps, simulate on the 8x8 torus, and rank candidates by the number u of
distinct previously unknown rule bits queried in their predicted paths.
The joint probability of that path is exactly 2^(-u) under this prior:
unqueried bits are marginalized and repeated queries do not multiply mass.
This is not independent pixel likelihood or a vote across candidates.
The first candidate fills unknown slots with zero, the second with one;
the others sample independently. All budgets are nested and use the same
candidate pool. A path is certified MAP only when its u equals the number
of distinct unknown slots necessarily queried in the first frame. The
remaining cases are approximate search, not certified MAP.

Two sets of 2048 validation worlds per seed, three seeds (42/43/44):
the previous validation data (seeds 30042..44) and fresh unseen-rule worlds
(seeds 40042..44). Every split is checked against the training rule identities
and the training data hash. The threshold, checkpoints and budgets were
fixed before fresh data evaluation. No future-dependent filtering.

## Results

Nonpersistent eight-step exact accuracy, mean across three seeds
(sample standard deviation, percentage points):

| Hypotheses | Previous validation | Fresh held-out |
|---:|---:|---:|
| 1 | 28.39 (1.50) | 28.49 (0.85) |
| 8 | 32.21 (1.28) | 32.47 (1.90) |
| 32 | 32.42 (1.23) | 32.87 (1.90) |
| 128 | 32.41 (1.25) | 32.85 (1.90) |

All 48 summary records cover 12,288 distinct validation trajectories
(24 records each split, including the identical known-group control).
Fresh-data mean posterior path mass grows from .29926 to .33883.
Actual exact accuracy saturates by 32 candidates; 128 improves predicted
mass slightly but not realized accuracy. Only about 22% of paths meet
the conservative MAP certificate on the original validation set.
This is a decoding improvement with unchanged learned weights, and is
not a claim that the latent world is identifiable from the prefix.
The previously reported 27.08% rank-2 soft-kernel greedy result uses a
different decoder; the 1-candidate control here isolates the search benefit.

## Reproduction

From repo root:

    PYTHONPATH=scripts python -m pytest -q scripts/tests/test_directional_search.py scripts/tests/test_directional_metric.py
    python scripts/run_directional_search.py --out NEW.jsonl
    python scripts/run_directional_search.py --data-seed-base 40000 --out FRESH.jsonl
    python scripts/plot_directional_search.py

Tests include independent complete-rule enumeration on small synthetic
examples, exact probability and certificate checks, persistent assignments,
prefix-only invariance, relabeling invariance, corner wrap and contradictory
class rejection. All 11 targeted tests passed; prior PR #13 was independently
rechecked (13 targeted tests, all 42 rollouts and 84 full-data BCE values).
No GPU job submitted: current authorized gateway health and challenge
requests both returned connection refused on localhost:18766.
No original R0, BPB, or manuscript figure files changed.
