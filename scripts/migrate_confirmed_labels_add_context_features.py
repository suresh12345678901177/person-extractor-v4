"""
scripts/migrate_confirmed_labels_add_context_features.py
============================================================
One-time (idempotent) migration: appends the 3 new context-window
features (preceding_context_is_person_cue, following_context_is_nonperson_cue,
is_isolated_line - see feature_extractor.py) to every EXISTING record in
datasets/feedback/confirmed_labels.jsonl, whose stored `features` arrays
were frozen at the old 16-field schema before these features existed.

Does NOT recompute the original 16 values (those are untouched - they
don't depend on document context, only on the candidate's own text/
detections, which this script doesn't have access to; only the 3 new
context features are computed here, from the record's source_file +
location offsets, and appended).

Skips (leaves untouched, reported separately) any record whose source
file no longer exists on disk or whose stored offset no longer points
at the expected text - never guesses. Padding with 0 (rather than
dropping) for a handful of records whose source file is gone: 0 means
"no adjacent word found", a safe, real value elsewhere in this same
feature (e.g. an isolated candidate genuinely has none), not a fabricated
guess about content.

Usage:
    python scripts/migrate_confirmed_labels_add_context_features.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.document_factory import DocumentFactory
from src.feedback.feedback_store import _read_jsonl_lines
from src.features.feature_extractor import _NONPERSON_CONTEXT_CUES, _PERSON_CONTEXT_CUES
from src.preprocessing.cleaner import clean_text
from src.preprocessing.segmenter import following_words, preceding_words

_LOC_RE = re.compile(r"chars (\d+)-(\d+)")
EXPECTED_OLD_LEN = 16


def _compute_new_features(document_text: str, start: int, end: int) -> list[int]:
    prev_words = [w.lower() for w in preceding_words(document_text, start, n=3)]
    preceding_is_person_cue = any(w in _PERSON_CONTEXT_CUES for w in prev_words)

    next_words = [w.lower() for w in following_words(document_text, end, n=3)]
    following_is_nonperson_cue = any(w.rstrip(".,") in _NONPERSON_CONTEXT_CUES for w in next_words)

    is_isolated = (not prev_words) and (not next_words)
    return [int(preceding_is_person_cue), int(following_is_nonperson_cue), int(is_isolated)]


def main() -> None:
    path = BASE_DIR / "datasets" / "feedback" / "confirmed_labels.jsonl"
    # _read_jsonl_lines splits on the literal newline byte only, not
    # str.splitlines() - context_text (added 2026-09-18, after this
    # script was originally written) carries real document text that can
    # legitimately contain Unicode line-separator-like characters
    # splitlines() would misparse as line breaks. See its docstring.
    records = [json.loads(line) for line in _read_jsonl_lines(path)]

    factory = DocumentFactory()
    cleaned_cache: dict[str, str | None] = {}

    already_migrated = 0
    updated = 0
    mismatched: list[tuple[str, str, str]] = []
    padded_missing_file: list[str] = []

    for record in records:
        if len(record["features"]) != EXPECTED_OLD_LEN:
            already_migrated += 1
            continue

        src = record["source_file"]
        match = _LOC_RE.search(record.get("location", ""))
        if not match:
            record["features"] = record["features"] + [0, 0, 0]
            padded_missing_file.append(record["feedback_id"])
            continue
        start, end = int(match.group(1)), int(match.group(2))

        if src not in cleaned_cache:
            p = Path(src)
            if not p.exists():
                cleaned_cache[src] = None
            else:
                result = factory.create(str(p))
                cleaned_cache[src] = clean_text(result.document.full_text) if result.success else None
        cleaned = cleaned_cache[src]

        if cleaned is None:
            record["features"] = record["features"] + [0, 0, 0]
            padded_missing_file.append(record["feedback_id"])
            continue

        actual = cleaned[start:end]
        if actual != record["text"]:
            mismatched.append((record["feedback_id"], record["text"], actual))
            record["features"] = record["features"] + [0, 0, 0]
            continue

        record["features"] = record["features"] + _compute_new_features(cleaned, start, end)
        updated += 1

    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total records:              {len(records)}")
    print(f"Already migrated (skipped): {already_migrated}")
    print(f"Updated with real context:  {updated}")
    print(f"Padded with [0,0,0] (source file missing): {len(padded_missing_file)}")
    print(f"Mismatched offset (padded with [0,0,0], not guessed): {len(mismatched)}")
    if mismatched:
        print("\nMismatched records (investigate before trusting their new features):")
        for fid, expected, actual in mismatched[:10]:
            print(f"  {fid}: expected {expected!r}, found {actual!r} at recorded offset")


if __name__ == "__main__":
    main()
