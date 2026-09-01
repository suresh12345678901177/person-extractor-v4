"""
scripts/auto_label_from_gold.py
==================================
Auto-confirms pending feedback items whose source file is one of our own
benchmark documents (datasets/benchmark/*.txt) against that file's
existing hand-built gold labels (*_gold.json), instead of asking a human
to re-judge a name we already have an authoritative answer for.

Matched by filename (not full path), since a benchmark file may have been
run from a different location (e.g. a Downloads copy) and still refer to
the same document this repo has gold labels for. A pending item's char
span is pulled out of its `location` string ("... (chars N-M)") and
matched against gold mention spans using the SAME overlap rule
src/evaluation/metrics.py uses for real benchmark scoring (any overlap
counts as a match) - so this produces the same answer --evaluate would,
just without a human re-reading each one.

Usage:
    python scripts/auto_label_from_gold.py            (apply)
    python scripts/auto_label_from_gold.py --dry-run   (preview only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.feedback.feedback_store import FeedbackStore

LOCATION_CHARS_RE = re.compile(r"chars (\d+)-(\d+)")


def _load_gold_spans_by_filename(benchmark_dir: Path) -> dict[str, list[tuple[int, int]]]:
    spans_by_name: dict[str, list[tuple[int, int]]] = {}
    for gold_path in benchmark_dir.glob("*_gold.json"):
        txt_name = gold_path.name.replace("_gold.json", ".txt")
        data = json.loads(gold_path.read_text(encoding="utf-8"))
        spans_by_name[txt_name] = [(m["start"], m["end"]) for m in data["mentions"]]
    return spans_by_name


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(not (end <= gs or start >= ge) for gs, ge in spans)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label pending feedback sourced from benchmark files against gold labels")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing any changes")
    args = parser.parse_args()

    store = FeedbackStore(BASE_DIR / "datasets" / "feedback")
    gold_spans_by_name = _load_gold_spans_by_filename(BASE_DIR / "datasets" / "benchmark")

    pending = store.load_pending()
    remaining_by_id = {r.feedback_id: r for r in pending}

    confirmed_person = 0
    confirmed_not_person = 0
    skipped_unmatched_location = 0
    to_write: list = []

    for record in pending:
        filename = Path(record.source_file).name
        gold_spans = gold_spans_by_name.get(filename)
        if gold_spans is None:
            continue  # not one of our benchmark files - leave for manual labeling

        m = LOCATION_CHARS_RE.search(record.location)
        if not m:
            skipped_unmatched_location += 1
            continue

        start, end = int(m.group(1)), int(m.group(2))
        is_person = _overlaps_any(start, end, gold_spans)

        record.status = "confirmed_person" if is_person else "confirmed_not_person"
        record.labeled_at_utc = datetime.now(timezone.utc).isoformat()
        to_write.append(record)
        remaining_by_id.pop(record.feedback_id, None)

        if is_person:
            confirmed_person += 1
        else:
            confirmed_not_person += 1

    print(f"Matched against gold: {len(to_write)} "
          f"({confirmed_person} person / {confirmed_not_person} not-person)")
    if skipped_unmatched_location:
        print(f"Skipped (couldn't parse location): {skipped_unmatched_location}")
    print(f"Remaining for manual labeling: {len(remaining_by_id)}")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    for record in to_write:
        store.append_confirmed(record)
    store.save_all_pending(list(remaining_by_id.values()))
    print("\nWrote confirmed labels and updated pending queue.")
    print("Run this next to fold them into the classifier:")
    print("  python scripts/retrain_from_feedback.py")


if __name__ == "__main__":
    main()
