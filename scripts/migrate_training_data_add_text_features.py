"""
scripts/migrate_training_data_add_text_features.py
=====================================================
One-time (idempotent) migration: appends the 3 text-only features added
2026-09-24 (common_word_ratio, first_token_is_ambiguous,
place_or_org_token_count - see FeatureVector) to every stored training
row whose `features` array still has the previous 19-field schema:

  datasets/training/synthetic_training_data.jsonl
  datasets/feedback/confirmed_labels.jsonl
  datasets/feedback/pending_review.jsonl

Unlike the context-feature migration before it, nothing here needs the
source document: these features depend on the candidate text alone, and
are computed by the SAME text_only_features() the live pipeline calls -
so every migrated row gets the exact value inference would produce, with
no padding or guessing. The original 19 values are never touched.

Each file is copied to a timestamped backup before being rewritten.

Usage:
    python scripts/migrate_training_data_add_text_features.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.feedback.feedback_store import _read_jsonl_lines
from src.features.feature_extractor import text_only_features
from src.features.feature_vector import FEATURE_NAMES
from src.knowledge.knowledge_base import KnowledgeBase

OLD_LEN = 19
NEW_LEN = len(FEATURE_NAMES)

TARGETS = [
    BASE_DIR / "datasets" / "training" / "synthetic_training_data.jsonl",
    BASE_DIR / "datasets" / "feedback" / "confirmed_labels.jsonl",
    BASE_DIR / "datasets" / "feedback" / "pending_review.jsonl",
]


def migrate_file(path: Path, kb: KnowledgeBase, stamp: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        print(f"{path.relative_to(BASE_DIR)}: empty or missing - nothing to do")
        return

    records = [json.loads(line) for line in _read_jsonl_lines(path)]
    updated = already = 0
    unexpected: list[int] = []
    for i, record in enumerate(records):
        n = len(record["features"])
        if n == NEW_LEN:
            already += 1
        elif n == OLD_LEN:
            # Feedback rows store the aggregation key as normalized_text;
            # synthetic rows only have "text" (already the normalized form).
            text = record.get("normalized_text") or record["text"]
            record["features"] = record["features"] + list(text_only_features(text, kb))
            updated += 1
        else:
            unexpected.append(i)

    if unexpected:
        print(f"{path.relative_to(BASE_DIR)}: {len(unexpected)} row(s) have neither {OLD_LEN} nor "
              f"{NEW_LEN} features (first at row {unexpected[0]}) - refusing to touch this file.")
        return
    if not updated:
        print(f"{path.relative_to(BASE_DIR)}: all {already} row(s) already migrated")
        return

    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.stem}_pre_text_features_{stamp}{path.suffix}"
    shutil.copy2(path, backup)

    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(BASE_DIR)}: migrated {updated}, already migrated {already} "
          f"(backup: {backup.relative_to(BASE_DIR)})")


def main() -> None:
    kb = KnowledgeBase.load(BASE_DIR / "assets")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in TARGETS:
        migrate_file(path, kb, stamp)


if __name__ == "__main__":
    main()
