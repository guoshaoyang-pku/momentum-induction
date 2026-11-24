# Raw-neighborhood evidence world model (2026-11-24)

An experimental CA model, **not R0**. A shared 9→16→8 MLP embeds each
of the 512 raw 3×3 binary patterns. No supplied neighbor counts or hidden
rule bits. Match observed situation/outcome pairs via a learned RBF kernel;
all evidence comes from frames 0..7. The 298-parameter mean arm averages
signed observations. The 299-parameter count arm instead sums them, learns
a symmetric next-cell noise floor, and uses a frozen inferred rule table
for true 8-frame closed-loop prediction. Shifted queries provide a matched
negative control. Toroidal indexing is explicit.

**45 independent CPU fits**: 33 at 600 minimum updates; 12 at 2400.
Original L3 self-consistent train stream, seeds 42/43/44, M=128/512/2048.
First-frame train BCE is over all 64M bits, validation BCE over the first
256 pinned worlds, closed-loop accuracy over all 2048 L3 worlds. L1
transfer evaluation over all 512 released GoL worlds (no L1 fine-tuning).
Separate unfiltered noisy worlds use independent flips at every transition,
including history. The same data hashes match the earlier Structance
comparison at all common anchors. No noise labels supplied to model.
All runs are finite-budget, not converged Structance estimates.

Results: mean arm achieves 100% closed-loop L3 SeqAcc for every seed and
size. Count arm reduces validation BCE and with 2400 updates at M=512
also achieves 100% SeqAcc for all three seeds. Shifted control remains
below 5% overall and about 1.2% on nonpersistent worlds. L1 performance
is not identical across models: evidence coverage is incomplete on
unfiltered worlds. Noise p=.1: the longer count run learns noise floor
near .1; p=.5: it instead reduces evidence gain, not a calibrated estimate
of p. Greedy stochastic rollout exact-match is not a probability-quality
metric; compare validation BCE to H(p). No claim that the model learns
toroidal geometry without its explicitly supplied raw neighborhoods.

Reproduce from repository root:
```sh
PYTHONPATH=scripts python -m unittest discover -s scripts/tests -p test_raw_evidence.py
PYTHONPATH=scripts python scripts/run_raw_evidence_experiment.py \
  --l3 data/eval_corpora/l3_zs_sc_g8.npz \
  --l1 data/eval_corpora/l1_val_gol_g8.npz --out /tmp/ca600.jsonl
PYTHONPATH=scripts python scripts/run_raw_evidence_experiment.py \
  --l3 data/eval_corpora/l3_zs_sc_g8.npz \
  --l1 data/eval_corpora/l1_val_gol_g8.npz --out /tmp/ca2400.jsonl \
  --sizes 512 --steps 2400 --mode count
```

The new models are also exposed in the native training factory as
raw_evidence_mean / raw_evidence_count / raw_evidence_shift.
The factory's standard training target is the eight-frame likelihood;
the finite-N experiment script is the separate first-frame protocol.
The original CNN/Transformer training and release data are unchanged.
