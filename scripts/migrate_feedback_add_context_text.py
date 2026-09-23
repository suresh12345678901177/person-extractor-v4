"""
scripts/migrate_feedback_add_context_text.py
================================================
One-time (idempotent) backfill: populates the new `context_text` field
(the full source line containing each candidate - see
src.feedback.feedback_store.FeedbackRecord and
src.preprocessing.segmenter.line_containing) for every EXISTING record in
both datasets/feedback/confirmed_labels.jsonl and
datasets/feedback/pending_review.jsonl whose `context_text` is missing or
empty (i.e. logged before this field existed).

Modeled directly on scripts/migrate_confirmed_labels_add_context_features.py's
verify-before-trust pattern: never guesses. Reopens record["source_file"],
parses the stored `location` string ("chars N-M") to recover offsets (old
records have no raw ints stored), verifies cleaned[start:end] ==
record["text"] before trusting the offset, and falls back to an explicit,
distinct placeholder (never a fabricated guess, never a bare "") when the
location string can't be parsed, the source file is gone/unreadable, or the
text no longer matches at that offset.

Writes a timestamped backup of each file to datasets/feedback/backups/
before overwriting it.

Usage:
    python scripts/migrate_feedback_add_context_text.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.document_factory import DocumentFactory
from src.feedback.feedback_store import _read_jsonl_lines
from src.preprocessing.cleaner import clean_text
from src.preprocessing.segmenter import line_containing

_LOC_RE = re.compile(r"chars (\d+)-(\d+)")

PLACEHOLDER_UNPARSEABLE = "[context unavailable: could not parse location]"
PLACEHOLDER_MISSING_SOURCE = "[context unavailable: source file no longer exists]"
PLACEHOLDER_MISMATCH = "[context unavailable: offset mismatch]"


def _migrate_file(path: Path, factory: DocumentFactory, cleaned_cache: dict[str, str | None]) -> None:
    print(f"=== {path.name} ===")
    if not path.exists():
        print("File does not exist - skipping.\n")
        return

    # _read_jsonl_lines splits on the literal newline byte only, not
    # str.splitlines() - see its docstring: context_text now carries real
    # (sometimes OCR'd) document text that can legitimately contain
    # Unicode line-separator-like characters (NEL, LS, PS) which
    # splitlines() would misparse as line breaks.
    records = [json.loads(line) for line in _read_jsonl_lines(path)]

    already_migrated = 0
    updated = 0
    padded_missing_file: list[str] = []
    padded_unparseable: list[str] = []
    mismatched: list[tuple[str, str, str]] = []

    for record in records:
        if record.get("context_text"):
            already_migrated += 1
            continue

        match = _LOC_RE.search(record.get("location", ""))
        if not match:
            record["context_text"] = PLACEHOLDER_UNPARSEABLE
            padded_unparseable.append(record["feedback_id"])
            continue
        start, end = int(match.group(1)), int(match.group(2))

        src = record["source_file"]
        if src not in cleaned_cache:
            p = Path(src)
            if not p.exists():
                cleaned_cache[src] = None
            else:
                result = factory.create(str(p))
                cleaned_cache[src] = clean_text(result.document.full_text) if result.success else None
        cleaned = cleaned_cache[src]

        if cleaned is None:
            record["context_text"] = PLACEHOLDER_MISSING_SOURCE
            padded_missing_file.append(record["feedback_id"])
            continue

        actual = cleaned[start:end]
        if actual != record["text"]:
            mismatched.append((record["feedback_id"], record["text"], actual))
            record["context_text"] = PLACEHOLDER_MISMATCH
            continue

        record["context_text"] = line_containing(cleaned, start)
        updated += 1

    if records:
        backup_dir = BASE_DIR / "datasets" / "feedback" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = path.stem
        backup_path = backup_dir / f"{stem}_pre_context_text_{timestamp}.jsonl"
        shutil.copy2(path, backup_path)

        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        backup_path = None

    print(f"Total records:                 {len(records)}")
    print(f"Already migrated (skipped):    {already_migrated}")
    print(f"Updated with real context:     {updated}")
    print(f"Padded (source file missing):  {len(padded_missing_file)}")
    print(f"Padded (unparseable location): {len(padded_unparseable)}")
    print(f"Padded (offset mismatch):      {len(mismatched)}")
    if backup_path:
        print(f"Backup written to:             {backup_path}")
    if mismatched:
        print("\nMismatched records (investigate before trusting their context_text):")
        for fid, expected, actual in mismatched[:10]:
            print(f"  {fid}: expected {expected!r}, found {actual!r} at recorded offset")
    print()


def main() -> None:
    feedback_dir = BASE_DIR / "datasets" / "feedback"
    factory = DocumentFactory()
    cleaned_cache: dict[str, str | None] = {}

    _migrate_file(feedback_dir / "confirmed_labels.jsonl", factory, cleaned_cache)
    _migrate_file(feedback_dir / "pending_review.jsonl", factory, cleaned_cache)


if __name__ == "__main__":
    main()
