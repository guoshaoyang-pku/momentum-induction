"""Generate / verify the 50-step all-default loss-sequence goldens for the
three models (constructive, art, diffusion) on CPU.

Usage:
  python3 scripts/tests/golden_bitcompat.py write   # produce goldens (current HEAD)
  python3 scripts/tests/golden_bitcompat.py check   # verify new code matches

The golden is the exact per-step loss string sequence (repr of float) for
steps 1..50 with every recipe-v2 flag at its default. Bit-compat requires
tolerance 0: the loss sequence must be identical, because with all flags
default the RNG consumption and compute path are unchanged.

This script runs the REAL train.py main() as a subprocess so the golden
covers the full training path (model build, data stream, optimizer, loss),
not a re-implementation.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, ".."))
GOLDEN_PATH = os.path.join(HERE, "golden_bitcompat.json")

# (model, extra-args) — all-default recipe (no recipe-v2 flags), tiny CPU run.
CASES = {
    "constructive": ["--model", "constructive", "--arm", "existence",
                     "--head", "concat"],
    "art": ["--model", "art", "--arm", "existence"],
    "diffusion": ["--model", "diffusion", "--arm", "existence"],
}

LOSS_RE = re.compile(r"^step\s+(\d+)\s+loss\s+(\S+)\s+tf-pixel")


def run_case(model, extra, workdir):
    cmd = [sys.executable, "-m", "cawm.train", "--task", "L1",
           "--steps", "50", "--batch", "16", "--lr", "1e-3", "--seed", "42",
           "--eval_every", "0", "--log_every", "1",
           "--out", f"gold_{model}"] + extra
    env = dict(os.environ, PYTHONPATH=SCRIPTS, OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run(cmd, cwd=workdir, env=env, check=True, capture_output=True)
    losses = {}
    log = os.path.join(workdir, "data", "runs", f"gold_{model}", "train.log")
    with open(log) as f:
        for line in f:
            m = LOSS_RE.match(line)
            if m:
                losses[int(m.group(1))] = m.group(2)
    return [losses[i] for i in range(1, 51)]


def main(mode, workdir):
    if mode == "write":
        gold = {m: run_case(m, e, workdir) for m, e in CASES.items()}
        with open(GOLDEN_PATH, "w") as f:
            json.dump(gold, f, indent=1)
        print(f"[golden] wrote {GOLDEN_PATH}")
        for m, seq in gold.items():
            print(f"  {m}: step1={seq[0]} step50={seq[-1]}")
    else:
        with open(GOLDEN_PATH) as f:
            gold = json.load(f)
        ok = True
        for m, e in CASES.items():
            got = run_case(m, e, workdir)
            want = gold[m]
            match = got == want
            ok &= match
            print(f"[bitcompat] {m}: {'IDENTICAL' if match else 'MISMATCH'}")
            if not match:
                for i, (g, w) in enumerate(zip(got, want), 1):
                    if g != w:
                        print(f"  first diff step {i}: got {g} want {w}")
                        break
        print("[bitcompat] " + ("ALL IDENTICAL" if ok else "FAILED"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    wd = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "_gold_tmp")
    os.makedirs(wd, exist_ok=True)
    main(sys.argv[1], wd)
