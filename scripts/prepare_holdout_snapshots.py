"""Create labelable TXT snapshots from a PDF/TXT holdout manifest.

Snapshots use the exact cleaning and PDF page joining used by Pipeline, so
gold offsets generated from them match the spans scored by `cli.py --evaluate`.
They can contain sensitive source text and are ignored by git.

Usage:
    python scripts/prepare_holdout_snapshots.py
    python scripts/prepare_holdout_snapshots.py --manifest path\\to\\holdout_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.document_factory import DocumentFactory
from src.core.models import SourceFormat
from src.preprocessing.cleaner import clean_text

PAGE_SEPARATOR = "\n\n"
DEFAULT_HOLDOUT_DIR = BASE_DIR / "datasets" / "evaluation_holdout"
DEFAULT_MANIFEST = DEFAULT_HOLDOUT_DIR / "holdout_manifest.jsonl"


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "document"


def _pipeline_ready_text(document) -> str:
    if document.source_format == SourceFormat.PDF:
        return PAGE_SEPARATOR.join(clean_text(page.text) for page in document.pages)
    return clean_text(document.full_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PDF/TXT holdout snapshots for manual gold labeling")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Holdout JSONL manifest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HOLDOUT_DIR / "snapshots",
                        help="Directory for TXT snapshots and gold labels")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing snapshots")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.exists():
        parser.error(f"Holdout manifest not found: {manifest}")

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").split("\n") if line.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = DocumentFactory()
    prepared = 0
    skipped = 0

    for index, row in enumerate(rows, start=1):
        source = Path(row["source_file"])
        result = factory.create(source)
        if not result.success or result.document is None:
            print(f"SKIPPED {source}: {result.error or 'unable to read'}")
            skipped += 1
            continue
        name = f"{index:03d}_{_safe_stem(source.stem)}_{row['sha256'][:10]}.txt"
        target = output_dir / name
        if target.exists() and not args.overwrite:
            print(f"SKIPPED {target.name}: already exists (use --overwrite to replace)")
            skipped += 1
            continue
        target.write_text(_pipeline_ready_text(result.document), encoding="utf-8")
        prepared += 1
        print(f"Wrote {target}")

    print(f"Prepared {prepared} snapshot(s); skipped {skipped}.")
    print("Copy one snapshot to a .md file, wrap every person mention in **double asterisks**, then run:")
    print("  python scripts/build_benchmark_from_markup.py marked.md file_name --output-dir datasets/evaluation_holdout/snapshots")
    print("After every snapshot has a matching *_gold.json, run:")
    print("  python cli.py --evaluate --benchmark-dir datasets/evaluation_holdout/snapshots")


if __name__ == "__main__":
    main()
