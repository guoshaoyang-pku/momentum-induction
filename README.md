# Cellular automata world models: code and results supplement

This package accompanies *Why Do Transformers and Diffusion Models Fail to Learn Cellular Automata?* It contains the CA simulator and models, 34 checksum-verified evaluation corpora, an offline result gallery, seed-level tables, and six selected E21 checkpoints. All commands below run from the extracted ZIP root.

## See the results

Open results/RESULTS.html in a browser. The page has nine selected paper figures, PNG previews, and vector PDFs. results/FIGURES.json records the image hashes. The included tables are:

- results/data/e11_same_weights_8_calls.csv: same-denoiser joint versus frame-by-frame sampling at eight network calls, seeds 42/43/44, on 8 x 8 Game of Life and 16 x 16 billiards. This is a seed-level summary; the E11 checkpoints and per-world predictions are not bundled.
- results/data/e21_selected_weights.csv and results/data/e21_depth_summary.json: strict L4B four-read checkpoint metrics and the full attention-depth summary.
- results/data/e22_rows.csv: 24 audited raster-transformer position/tokenizer cells.
- results/data/lifegpt_seeds.csv and results/data/e20_lifegpt_l4_seeds.csv: faithful raster LifeGPT seed data.

The current E27 full main matrix is absent because its audit has not completed. Figure scores are from fresh post-submission hash-split runs, not legacy submission numbers. The E11 comparison gives 42.25% joint and 99.93% frame-by-frame Game of Life SeqAcc across three seeds. The two samplers use the same trained weights within each seed and eight network calls per rollout; the comparison changes sampling order, noise updates, and commitment together.

## Environment and quick checks

Use Python 3.10–3.13. The local CPU verification environment was Python 3.10.11, NumPy 2.2.6, PyTorch 2.13.0, and pytest 9.1.1. `requirements.txt` pins the three package versions. Install them in a fresh environment from the extracted ZIP root:

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt

The listed PyTorch wheel resolves from public PyPI for macOS arm64 on Python 3.10 and 3.13; Linux x86_64 has a corresponding wheel. These resolution checks do not establish runtime behavior on every platform. The registered GPU runs used CUDA and may need a platform-specific PyTorch installation. The gallery opens without Python, TeX, or network access.

    python3 scripts/release_manifest.py --check
    PYTHONPATH=scripts python3 -m cawm.train --help
    PYTHONPATH=scripts python3 -m pytest scripts/tests
    PYTHONPATH=scripts python3 -m cawm.train --task L1 --model art --arm emergence --grid 8 --seed 42 --steps 200 --batch 8 --device cpu --no-bf16 --no-bf16-eval --eval_every 0 --out l1_smoke

The manifest check verifies every shipped file and all corpus sidecar hashes. The 200-step run checks code execution, not paper performance. It writes data/runs/ and data/ckpt/ in the extracted directory. The full staged test suite has two known exact-golden constructive loss mismatches at four-decimal precision around steps 45/47/50; its other tests passed in the latest full run. The golden fixtures remain unchanged.

## Task and data protocol

Each CA rule is an 18-bit outer-totalistic transition table. A permutation seeded by 42 splits all 262,144 rules into disjoint training and zero-shot halves. The main CA task uses an 8 x 8 toroidal grid, eight observed frames, and eight future frames. The separate billiards comparison uses 16 x 16 grids. The training stream is index-addressed with data seed 42; the model seed is separate. L1 is a fixed Game of Life rule. L23 trains on training-half rules: L2 evaluates new trajectories from that half, and L3 evaluates held-out rules. L4 adds a specified family constraint and tests a needed inference when direct evidence is silent.

Self-consistent corpora exclude queries that the observations and registered constraint cannot answer. The primary L1 corpus has 512 trajectories. L2 and L3 primary corpora have 2,048 each. The strict L4B validation and test corpora each have 768 trajectories that exercise the relational constraint; the test is on held-out rules. Every supplied .npz has a .npz.json sidecar with its SHA-256 and sampling seed. The label for these corpora is post-submission regenerated (seed=42 rule split), hash-split protocol, grid 8.

## Evaluate the bundled checkpoints

The six weights/ files are final E21 EMA checkpoints: three seeds of the ordinary-loss 3D-convolution pairing model and three seeds of the KV-shift model with a training-only evidence-balanced loss. Each checkpoint hash is in results/data/e21_selected_weights.csv. The task-aware evidence labels were never model inputs. Run the following on the strict 768-world L4B test corpus, then repeat with each checkpoint filename listed in the CSV:

    PYTHONPATH=scripts python3 scripts/diagnose_structured.py --checkpoint weights/e21_art_L4b_ex10_r4_s42.pt --corpus l4b_zs_scx_g8 --device cpu --batch 64 --out /tmp/e21_art_s42_eval.json

Compare metrics.seq_acc in the generated JSON with the CSV. The bundled ART seed-42 weight reproduced 1.0 SeqAcc on CPU during release preparation. Different kernels can change borderline predictions, so record hardware and library versions for a rerun. The checkpoint contains the original training args, optimizer state, and EMA weights. Only final EMA weights are used for this evaluation.

## Re-run selected training arms

Use PYTHONPATH=scripts python3 -m cawm.train as the single training entry point. E21 used 40,000 steps, batch 512, Adam at learning rate 0.001, cosine schedule with 500 warmup steps, clip 1, EMA 0.999, a self-consistent stream, bf16 training, and fp32 final evaluation. For the seed-42 arms:

    PYTHONPATH=scripts python3 -m cawm.train --task L4 --model art --arm emergence --head concat --grid 8 --l4v2 l4b --ex_pool data/eval_corpora/l4b_train_expool_g8.npz --ex_frac 0.10 --art_reads 4 --seed 42 --steps 40000 --batch 512 --workers 4 --lr 0.001 --clip 1 --lr_schedule cosine --warmup 500 --ema 0.999 --sc_stream --bf16 --no-bf16-eval --eval_every 0 --log_every 500 --save_every 10000 --device cuda --out e21_art_L4b_ex10_r4_s42
    PYTHONPATH=scripts python3 -m cawm.train --task L4 --model art_kvshift --arm emergence --head concat --grid 8 --l4v2 l4b --ex_pool data/eval_corpora/l4b_train_expool_g8.npz --ex_frac 0.10 --kv_reads 4 --evidence_loss slot_relation --seed 42 --steps 40000 --batch 512 --workers 4 --lr 0.001 --clip 1 --lr_schedule cosine --warmup 500 --ema 0.999 --sc_stream --bf16 --no-bf16-eval --eval_every 0 --log_every 500 --save_every 10000 --device cuda --out e21_kv_L4b_ex10_sr_r4_s42

Change seed to 43 and 44 for the remaining selected arms. E21 chose read depth on the training-half validation corpus before scoring the held-out strict test. Full protocol reproduction also needs the other read depths, losses, L1/L23 controls, and all registered seeds; the selected arms alone do not recreate the study.

E22 used a separately frozen code variant at variants/e22/ (original cawm fingerprint e54b5480ece8a813). It tested widths 64/128, cell/pair tokenizers, 1D/axial RoPE, and seeds 42/43/44 at 40,000 steps and batch 512. Run its commands from variants/e22/ with PYTHONPATH=scripts. The bundled E22 table permits reaggregation, but E22 trained weights and raw predictions are not in this ZIP. The primary code is the E21 variant (original cawm fingerprint fb949ed04d618ee2). Private prose/path scrubbing changed the published source hashes; use the registered original fingerprints for provenance and the bundled manifest for integrity.

## Scope

The supplied six weights and pinned corpus permit re-evaluating a complete three-seed slice of the E21 L4B result. The tables and gallery let readers inspect selected reported outcomes. A full paper reproduction needs suitable GPUs and all registered training and evaluation arms, including negative results. No license is asserted here; redistribution terms must be supplied by the authors before a public source-code release.

For a dependency-free check of the bundled summaries, weights, and figure hashes, run python3 scripts/verify_results.py. requirements.txt records the observed CPU test environment. python3 scripts/make_zip.py checks the manifest and creates a deterministic ZIP beside this directory.
