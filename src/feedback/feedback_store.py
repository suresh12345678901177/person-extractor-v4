"""
src.feedback.feedback_store
==============================
Persists every REVIEW-bucket candidate (the ones the rules engine
correctly identifies as plausible but can't auto-confirm - usually a
real name just not in the knowledge base dictionary yet) to a JSONL log,
so a human can later confirm or correct them and those real corrections
can retrain the classifier. This is the mechanism behind the active
learning loop: the system's blind spots become visible and fixable
instead of silently repeating on every future document.

File format: one JSON object per line, in `datasets/feedback/`:
  pending_review.jsonl     - awaiting a human decision
  confirmed_labels.jsonl   - human-confirmed (label=1 or label=0),
                             ready for scripts/retrain_from_feedback.py
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.core.models import AggregatedPerson
from src.utils.logger import get_logger

logger = get_logger("feedback.feedback_store")

PENDING_FILENAME = "pending_review.jsonl"
CONFIRMED_FILENAME = "confirmed_labels.jsonl"


@dataclass(slots=True)
class FeedbackRecord:
    feedback_id: str
    source_file: str
    text: str
    normalized_text: str
    location: str
    confidence: float
    features: list[float]
    logged_at_utc: str
    status: str = "pending"  # pending | confirmed_person | confirmed_not_person | skipped
    labeled_at_utc: str | None = None

    def as_dict(self) -> dict:
        return {
            "feedback_id": self.feedback_id,
            "source_file": self.source_file,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "location": self.location,
            "confidence": self.confidence,
            "features": self.features,
            "logged_at_utc": self.logged_at_utc,
            "status": self.status,
            "labeled_at_utc": self.labeled_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackRecord":
        return cls(**d)


class FeedbackStore:
    def __init__(self, feedback_dir: str | Path = "datasets/feedback") -> None:
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.feedback_dir / PENDING_FILENAME
        self.confirmed_path = self.feedback_dir / CONFIRMED_FILENAME

    def log_review_persons(self, persons: list[AggregatedPerson], source_file: str) -> int:
        """Append every REVIEW-bucket person's first mention to the
        pending feedback log. Deduplicates against text already pending
        or already confirmed, so re-running on the same file doesn't
        pile up duplicate labeling work."""
        already_seen = self._existing_texts()

        new_records: list[FeedbackRecord] = []
        for person in persons:
            key = person.normalized_text.lower()
            if key in already_seen:
                continue
            first_mention = person.mentions[0]
            if first_mention.state.feature_vector is None:
                continue  # nothing to train on without features

            record = FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                source_file=source_file,
                text=person.display_text,
                normalized_text=person.normalized_text,
                location=str(first_mention.candidate.location) if first_mention.candidate.location else "",
                confidence=first_mention.state.final_confidence,
                features=first_mention.state.feature_vector.as_list(),
                logged_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            new_records.append(record)
            already_seen.add(key)

        if new_records:
            with self.pending_path.open("a", encoding="utf-8") as fh:
                for r in new_records:
                    fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
            logger.info("Logged %d new candidate(s) to %s for review", len(new_records), self.pending_path)

        return len(new_records)

    def load_pending(self) -> list[FeedbackRecord]:
        if not self.pending_path.exists():
            return []
        records = []
        for line in self.pending_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(FeedbackRecord.from_dict(json.loads(line)))
        return [r for r in records if r.status == "pending"]

    def save_all_pending(self, records: list[FeedbackRecord]) -> None:
        """Overwrite the pending file with the given records (used after
        a labeling session to persist status updates)."""
        with self.pending_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")

    def append_confirmed(self, record: FeedbackRecord) -> None:
        with self.confirmed_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    def load_confirmed(self) -> list[FeedbackRecord]:
        if not self.confirmed_path.exists():
            return []
        records = []
        for line in self.confirmed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(FeedbackRecord.from_dict(json.loads(line)))
        return records

    def _existing_texts(self) -> set[str]:
        seen = set()
        for r in self.load_pending():
            seen.add(r.normalized_text.lower())
        for r in self.load_confirmed():
            seen.add(r.normalized_text.lower())
        return seen
