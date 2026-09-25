"""Create a deterministic, file-separated PDF/TXT evaluation manifest.

The manifest stores paths and hashes only; it never copies case material.
Files listed here are automatically excluded by retrain_from_feedback.py,
so they remain an honest holdout set.

Usage:
    python scripts/create_pdf_txt_holdout.py --input-dir "C:\\case_files" --count 25
    python scripts/create_pdf_txt_holdout.py --input-dir "C:\\case_files" --count 40 --seed 20260925
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt"})
DEFAULT_MANIFEST = BASE_DIR / "datasets" / "evaluation_holdout" / "holdout_manifest.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PDF/TXT file-separated holdout manifest")
    parser.add_argument("--input-dir", required=True, help="Folder containing PDF and TXT files")
    parser.add_argument("--count", type=int, default=25, help="How many files to reserve (default: 25)")
    parser.add_argument("--seed", type=int, default=20260925, help="Sampling seed recorded in the manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST, help="Manifest JSONL path")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    root = Path(args.input_dir).resolve()
    if not root.is_dir():
        parser.error(f"--input-dir is not a directory: {root}")

    available = _candidate_files(root)
    if not available:
        parser.error("No .pdf or .txt files found under --input-dir")

    rng = random.Random(args.seed)
    selected = list(available)
    rng.shuffle(selected)
    selected = selected[: min(args.count, len(selected))]
    selected.sort(key=lambda path: str(path).lower())

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in selected:
            stat = path.stat()
            row = {
                "source_file": str(path),
                "relative_path": str(path.relative_to(root)),
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": stat.st_size,
                "sha256": _sha256(path),
                "selection_seed": args.seed,
                "selected_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Reserved {len(selected)} of {len(available)} PDF/TXT files for evaluation.")
    print(f"Manifest: {output}")
    print("Do not label these files for retraining. Label their complete person mentions separately for evaluation.")


if __name__ == "__main__":
    main()
