# Learned local geometry under a changed CA rule family

This is a synthetic non-R0 control, not a change to the original L3 data,
architecture, or Structance benchmark. It asks whether the learned 20-parameter
metric really adapts to a new rule geometry rather than merely replaying the
original outer-totalistic rule slots.

The generator uses 50 independent binary rule entries indexed by center
state, number of active axial neighbors (0..4), and number of active
diagonal neighbors (0..4). All transitions use an 8x8 torus. There is no
future-dependent self-consistency filtering. Each world has a persistent
rule. The initial density is uniform from .1 to .9. This is NOT the original
L3 rule distribution. Train/validation rule identities are disjoint
(checked explicitly), and all models of the same seed and M see identical
worlds. The same 2048 validation worlds per seed are shared across sizes.
Training sets across M are independently sized, not nested.

Each learned model sees the nine raw neighborhood bits and observed
prefix transitions only, with no group or rule labels. It learns a rank-r
linear metric, a positive scale, and a bias, using first-frame BCE only.
The frozen control uses the previously fitted outer-totalistic learned
metric from Structance PR #34 (6307e89), with only its two calibration
parameters refitted. This is not the original R0 CNN/Transformer.
A shifted-query control retains the same model but breaks spatial alignment.
A privileged known-group oracle uses exact group evidence, a .5 probability
for unseen groups, and zero for greedy ties; this is not an optimal joint
trajectory predictor. Full-data BCE weights all 64M target bits equally.
Rollout uses the fixed prefix-derived table and its own predictions,
without true future frames. Nonpersistent means at least one future frame
differs from the last observed frame.

## Results

42 records, including 33 actual Adam fits and 9 oracle controls. Three
seeds (42/43/44), M=128/512/2048, 800/400/800 updates, batch 32,
lr .01, CPU. These finite budgets differ by M; no convergence or Structance
slope claim is made.

M=512 three-seed means (sample SD in parentheses):

| Model | Trainable parameters | Validation BCE (nats/bit) | Nonpersistent SeqAcc8 (%) |
|---|---:|---:|---:|
| Known groups, oracle | 0 | .00937 (.00056) | 28.39 (1.50) |
| Frozen original metric | 2 (20 allocated) | .44566 (.00136) | .53 (.14) |
| Rank 1 | 11 | .23453 (.01767) | .66 (.03) |
| Rank 2 | 20 | .03410 (.00180) | 26.90 (.10) |
| Rank 3 | 29 | .03039 (.00187) | 26.36 (.93) |
| Rank 3 shifted | 29 | .66255 (.00060) | .23 (.03) |

M=2048 rank 2/3 reach .03167/.02651 BCE and 27.08/26.90% nonpersistent
SeqAcc8. Rank 2 can represent distinct axial and diagonal counts despite
having only two latent dimensions. Rank 3 improves marginal BCE but does not
improve greedy exact rollouts. The frozen original geometry confounds
equal-total-count configurations, while learning from the new distribution
recovers nearly the known-group greedy rollout. At M=512, mean cross-group
kernel similarity falls from .06066 frozen to .000576 for rank 2 and
.000231 for rank 3. Within-group similarity is .96059/.97286/.85776,
respectively. This is empirical separation, not a proof of minimal rank.

The known-group oracle itself scores only 28.39% on nonpersistent worlds
because unseen rule slots can be queried later; even perfect local rule
geometry cannot guarantee eight-step recovery on this unfiltered family.
This is separate from the earlier SC-filtered result.

## Reproduction and checks

The small frozen-weight archive is from Structance PR #34; supply its
path as ARCHIVE. Its SHA256 and learned coefficients are stored in rows.
No checkpoint or raw corpus is added to this repository.

    python scripts/run_directional_metric.py --archive ARCHIVE --out NEW.jsonl --steps 400
    python scripts/run_directional_metric.py --archive ARCHIVE --out NEW_RANK1.jsonl --steps 400 --models rank1
    python scripts/run_directional_metric.py --archive ARCHIVE --out NEW_M128.jsonl --M 128 --steps 800 --models exact_group_oracle,rank1,rank2,rank3
    python scripts/run_directional_metric.py --archive ARCHIVE --out NEW_M2048.jsonl --M 2048 --steps 800 --models exact_group_oracle,rank1,rank2,rank3
    PYTHONPATH=scripts python -m pytest -q scripts/tests/test_directional_metric.py scripts/tests/test_art_cached.py scripts/tests/test_art_sliding_cached.py
    python scripts/qa_directional_metric_results.py
    python scripts/plot_directional_metric.py

13 targeted tests passed. Independent NumPy float64 reconstruction of all
84 complete-data BCE measurements differs by at most 3.56e-8 nats/bit.
All 42 closed-loop records were reproduced using the original float32
arithmetic; one first-frame bit flips at a near-zero threshold in float64
(max logit difference 6.85e-5 across all rows), so exact greedy metrics are
explicitly float32. Indexed torus edges, shift control and future invariance
are tested. No GPU dispatched: the authorized local HTTPS gateway returned
connection refused on port 18766. Original R0 and BPB files are untouched.
