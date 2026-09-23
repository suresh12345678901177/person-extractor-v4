"""
scripts/strip_real_case_feedback_records.py
===============================================
One-time (idempotent) cleanup for independent-audit finding #8's
escalated scope (2026-09-23): removes every datasets/feedback/*.jsonl
record whose source_file points into a real, external casework folder
rather than this project's own benchmark/public-source text.

Background: confirmed_labels.jsonl is meant to ship with only
benchmark- and public-book-sourced entries (see README's "Known
limitations"), but nothing ever code-enforced that - see the
2026-09-23 root-cause trace in README's "Independent-audit findings"
section. A manual audit found 660 of 1,313 records (119
confirmed_person, 541 confirmed_not_person) were sourced from a real
case folder, identified here by an EXTERNAL_SOURCE_MARKERS match
against each record's `source_file` field (the reliable signal - unlike
scripts/check_feedback_privacy.py's content-pattern matching, which
only caught 5 of these 660 because it searched context_text content,
not the always-present source_file field).

This does not delete the removed records outright: it writes a
timestamped backup first (same convention as every other migration
script here), and the exact removed set is reported in full so it can
be reconstructed if ever needed. What it does NOT do: decide whether a
record's source is "sensitive enough" to matter - it removes by SOURCE
LOCATION only (case folder vs. not), a decision already made by the
project owner, not a judgment about content this script isn't
positioned to make.

Usage:
    python scripts/strip_real_case_feedback_records.py            (apply)
    python scripts/strip_real_case_feedback_records.py --dry-run  (preview only)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.feedback.feedback_store import _read_jsonl_lines

# Markers identifying a source_file as pointing into real external
# casework rather than this project's own benchmark/public-source text.
# Same case-folder fingerprint as
# assets/privacy_guard/case_identifier_patterns.txt, applied to
# source_file (always present, machine-readable) instead of
# context_text (free text, easy to miss a mention in).
EXTERNAL_SOURCE_MARKERS = ("ProDiscover",)


def _is_external_source(source_file: str) -> bool:
    return any(marker in source_file for marker in EXTERNAL_SOURCE_MARKERS)


def _process_file(path: Path, dry_run: bool) -> None:
    print(f"=== {path.name} ===")
    if not path.exists():
        print("File does not exist - skipping.\n")
        return

    records = [json.loads(line) for line in _read_jsonl_lines(path)]
    kept, removed = [], []
    for record in records:
        if _is_external_source(record.get("source_file", "")):
            removed.append(record)
        else:
            kept.append(record)

    print(f"Total records:   {len(records)}")
    print(f"Kept:            {len(kept)}")
    print(f"Removed:         {len(removed)}")
    if removed:
        by_status: dict[str, int] = {}
        for r in removed:
            by_status[r.get("status", "pending")] = by_status.get(r.get("status", "pending"), 0) + 1
        print(f"  Removed by status: {by_status}")

    if dry_run:
        print("--dry-run: no files written.\n")
        return

    if removed:
        backup_dir = BASE_DIR / "datasets" / "feedback" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"{path.stem}_pre_external_source_strip_{timestamp}.jsonl"
        shutil.copy2(path, backup_path)
        print(f"Backup written to: {backup_path}")

        with path.open("w", encoding="utf-8") as fh:
            for record in kept:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(kept)} record(s) back to {path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    feedback_dir = BASE_DIR / "datasets" / "feedback"
    _process_file(feedback_dir / "confirmed_labels.jsonl", args.dry_run)
    _process_file(feedback_dir / "pending_review.jsonl", args.dry_run)


if __name__ == "__main__":
    main()
