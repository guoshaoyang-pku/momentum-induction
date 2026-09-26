#!/usr/bin/env python3
"""Re-evaluate every bundled E21 checkpoint on the strict L4B test corpus."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from diagnose_structured import audit


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/data/e21_selected_weights.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(device: str, batch: int, output_dir: Path | None) -> list[dict]:
    rows = list(csv.DictReader(TABLE.open(newline="")))
    if len(rows) != 6 or {(r["model"], r["seed"]) for r in rows} != {
        (model, str(seed)) for model in ("art", "kv") for seed in (42, 43, 44)
    }:
        raise ValueError("E21 selection table must contain both models and seeds 42/43/44")
    records = []
    # The evaluator locates corpora relative to the extracted supplement root.
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix="cawm-e21-") if output_dir is None else contextlib.nullcontext(output_dir) as destination:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for row in rows:
            checkpoint = ROOT / row["checkpoint"]
            corpus = ROOT / "data/eval_corpora" / (row["corpus"] + ".npz")
            if sha256(checkpoint) != row["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            if sha256(corpus) != row["corpus_sha256"]:
                raise ValueError(f"corpus hash mismatch: {corpus}")
            output = destination / f"{row['model']}_seed{row['seed']}.json"
            with contextlib.redirect_stdout(io.StringIO()):
                report = audit(str(checkpoint), output, device=device, batch=batch, corpus_name=row["corpus"])
            observed = float(report["metrics"]["seq_acc"])
            expected = float(row["seq_acc"])
            record = {
                "model": row["model"], "seed": int(row["seed"]),
                "expected_seq_acc": expected, "observed_seq_acc": observed,
                "exact_match": observed == expected, "n": report["n"],
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, help="retain per-world predictions and audit JSON")
    args = parser.parse_args()
    records = reproduce(args.device, args.batch, args.output_dir)
    if not all(row["exact_match"] for row in records):
        raise SystemExit("E21 score mismatch; inspect runtime, predictions, and library versions")
    print("all six E21 scores matched the bundled table exactly")


if __name__ == "__main__":
    main()
