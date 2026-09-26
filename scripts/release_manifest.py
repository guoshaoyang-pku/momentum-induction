#!/usr/bin/env python3
"""Build and verify a public release manifest.

The release branch intentionally contains a small, curated tree rather than
the research checkout.  This script is dependency-free so it can be run before
installing PyTorch:

    python scripts/release_manifest.py --write
    python scripts/release_manifest.py --check

``--write`` records SHA-256 and byte size for every release file.  ``--check``
recomputes those hashes and also checks the hash sidecars shipped with pinned
``.npz`` corpora.  Both modes fail on private research paths or common local
host/identity leaks in text files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "RELEASE_MANIFEST.json"

# These names identify internal provenance, raw run history, or machine-local
# material.  A public release should fail closed if any of them is copied in.
FORBIDDEN_COMPONENTS = {
    "docs",
    "paper_legacy",
    "data/paper",
    "data/runs",
    "tmp",
    ".verdent",
    ".git_disabled",
}

# Keep this list intentionally conservative.  It catches paths that would be
# embarrassing in a supplement while avoiding false positives on normal model
# names and citations.
LEAK_PATTERNS = (
    re.compile(r"/Users/[^\s'\"]+"),
    re.compile(r"/data/(?:shared|home|guoshaoyang|tione)/[^\s'\"]*"),
    re.compile(r"(?:360-[12]|ophis-gpu|a100_(?:perm|t1))"),
    re.compile(r"(?:Overleaf|Feishu|doubao\.com|xwechat_files)", re.I),
    re.compile(r"[A-Za-z0-9._%+-]+@(?:pku|tsinghua|gmail)\.[A-Za-z.]+", re.I),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def forbidden(path: Path) -> str | None:
    name = rel(path)
    parts = set(path.relative_to(ROOT).parts)
    if parts & {x for x in FORBIDDEN_COMPONENTS if "/" not in x}:
        return "private/internal path component"
    for prefix in ("data/paper/", "data/runs/"):
        if name.startswith(prefix):
            return "private/raw research path"
    return None


def iter_files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        name = rel(path)
        if name == MANIFEST_NAME or name == ".git" or name.startswith(".git/"):
            continue
        # Finder metadata and Python caches are never release inputs.
        if path.name == ".DS_Store" or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        yield path


def text_leaks(path: Path) -> list[str]:
    # The verifier necessarily contains the detector regexes as examples.
    if path.name == "release_manifest.py":
        return []
    # Check only formats likely to contain prose/paths.  Binary model weights
    # and corpora are hashed but never decoded as text.
    if path.suffix.lower() not in {".md", ".txt", ".json", ".py", ".sh", ".toml", ".yaml", ".yml", ".ini", ".tex", ".csv"}:
        return []
    try:
        value = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [pattern.pattern for pattern in LEAK_PATTERNS if pattern.search(value)]


def corpus_sidecar_errors(files: set[str]) -> list[str]:
    errors: list[str] = []
    for npz_name in sorted(x for x in files if x.endswith(".npz")):
        sidecar = npz_name + ".json"
        if sidecar not in files:
            errors.append(f"missing corpus sidecar: {sidecar}")
            continue
        npz = ROOT / npz_name
        try:
            data = json.loads((ROOT / sidecar).read_text(encoding="utf-8"))
        except Exception as exc:  # malformed sidecars are release blockers
            errors.append(f"invalid corpus sidecar {sidecar}: {exc}")
            continue
        expected = data.get("sha256")
        actual = sha256(npz)
        if expected != actual:
            errors.append(f"corpus hash mismatch: {npz_name} ({actual} != {expected})")
    return errors


def collect() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    for path in iter_files():
        why = forbidden(path)
        if why:
            errors.append(f"{rel(path)}: {why}")
            continue
        leaks = text_leaks(path)
        if leaks:
            errors.append(f"{rel(path)}: possible private reference ({', '.join(leaks)})")
        records.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    names = {item["path"] for item in records}
    errors.extend(corpus_sidecar_errors(names))
    return records, errors


def load_manifest() -> dict:
    path = ROOT / MANIFEST_NAME
    if not path.exists():
        raise SystemExit(f"missing {MANIFEST_NAME}; run --write first")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {MANIFEST_NAME}: {exc}") from exc
    if value.get("manifest_version") != 1:
        raise SystemExit("unsupported manifest_version")
    return value


def check_manifest() -> int:
    manifest = load_manifest()
    records, errors = collect()
    expected = {item["path"]: item for item in manifest.get("files", [])}
    actual = {item["path"]: item for item in records}
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"manifest file missing or now excluded: {name}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"unrecorded release file: {name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name]["bytes"] != actual[name]["bytes"] or expected[name]["sha256"] != actual[name]["sha256"]:
            errors.append(f"hash changed: {name}")
    if errors:
        print("RELEASE CHECK FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"release check passed: {len(actual)} files, manifest={MANIFEST_NAME}")
    return 0


def write_manifest() -> int:
    records, errors = collect()
    if errors:
        print("RELEASE MANIFEST REFUSED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    source_date = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date:
        generated = datetime.fromtimestamp(int(source_date), timezone.utc).isoformat()
    else:
        generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "manifest_version": 1,
        "generated_utc": generated,
        "purpose": "public supplement release; curated code, pinned corpora, figures, and optional selected weights",
        "files": records,
    }
    (ROOT / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_NAME}: {len(records)} files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="build RELEASE_MANIFEST.json")
    mode.add_argument("--check", action="store_true", help="verify manifest, hashes, sidecars, and privacy rules")
    args = parser.parse_args()
    return write_manifest() if args.write else check_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
