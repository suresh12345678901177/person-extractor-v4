"""
scripts/label_feedback.py
============================
Interactive terminal tool to confirm or correct the REVIEW-bucket
candidates the pipeline has logged. This is the human half of the
active learning loop: run it periodically, answer a few quick
yes/no/skip questions, and your real corrections become training data
for scripts/retrain_from_feedback.py.

Usage:
    python scripts/label_feedback.py
    python scripts/label_feedback.py --limit 20   (label only the first 20)

For each item you can answer:
    y  - yes, this IS a real person name
    n  - no, this is NOT a person (organization, title, noise, etc.)
    s  - skip for now (leave pending, ask again next time)
    q  - quit and save progress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.feedback.feedback_store import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Label pending REVIEW-bucket candidates")
    parser.add_argument("--limit", type=int, default=None, help="Only label the first N pending items")
    args = parser.parse_args()

    store = FeedbackStore(BASE_DIR / "datasets" / "feedback")
    pending = store.load_pending()

    if not pending:
        print("No pending items to label. Run the pipeline on some documents first -")
        print("REVIEW-bucket candidates are logged automatically.")
        return

    if args.limit:
        pending = pending[:args.limit]

    print(f"\n{len(pending)} pending item(s) to review.")
    print("Answer: [y]es person / [n]o not a person / [s]kip / [q]uit\n")

    all_pending = store.load_pending()
    remaining_by_id = {r.feedback_id: r for r in all_pending}
    confirmed_count = 0
    corrected_count = 0

    for i, record in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] \"{record.text}\"")
        print(f"    location: {record.location}   confidence: {record.confidence:.2f}   source: {record.source_file}")
        if record.context_text:
            print(f"    context:  {record.context_text}")

        answer = input("    Is this a real person name? [y/n/s/q]: ").strip().lower()

        if answer == "q":
            break
        if answer == "s" or answer == "":
            continue
        if answer not in ("y", "n"):
            print("    (unrecognized answer, skipping)")
            continue

        record.status = "confirmed_person" if answer == "y" else "confirmed_not_person"
        from datetime import datetime, timezone
        record.labeled_at_utc = datetime.now(timezone.utc).isoformat()

        store.append_confirmed(record)
        remaining_by_id.pop(record.feedback_id, None)

        if answer == "y":
            confirmed_count += 1
        else:
            corrected_count += 1
        print()

    # Persist remaining (unlabeled + skipped) items back to the pending file
    still_pending = list(remaining_by_id.values())
    store.save_all_pending(still_pending)

    print("=" * 60)
    print(f"Confirmed as person:     {confirmed_count}")
    print(f"Corrected (not person):  {corrected_count}")
    print(f"Still pending:           {len(still_pending)}")
    print("=" * 60)
    if confirmed_count or corrected_count:
        print("\nRun this next to fold your corrections into the classifier:")
        print("  python scripts/retrain_from_feedback.py")


if __name__ == "__main__":
    main()
