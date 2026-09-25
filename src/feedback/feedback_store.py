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

from src.core.models import AggregatedPerson, CandidateResult
from src.preprocessing.segmenter import line_containing
from src.utils.logger import get_logger

logger = get_logger("feedback.feedback_store")

PENDING_FILENAME = "pending_review.jsonl"
CONFIRMED_FILENAME = "confirmed_labels.jsonl"


def _read_jsonl_lines(path: Path) -> list[str]:
    """Split a JSONL file into its records on the literal newline BYTE
    only - deliberately NOT str.splitlines(), which also treats Unicode
    line-separator-like characters (NEL U+0085, LS U+2028, PS U+2029,
    etc.) as line breaks. Those can legitimately appear raw inside a JSON
    string value (context_text is pulled from real, sometimes OCR'd
    casework documents) without needing escaping per the JSON spec -
    splitlines() would then fragment one valid JSON record into multiple
    unparseable pieces, exactly matching how these files are actually
    written (json.dumps(...) + "\\n" per record, see log_review_persons
    and append_confirmed below)."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


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
    context_text: str = ""  # full source line containing this candidate at
    # capture time (src.preprocessing.segmenter.line_containing). "" only
    # for records logged before this field existed - see
    # scripts/migrate_feedback_add_context_text.py, never silently guessed.
    status: str = "pending"  # pending | confirmed_person | confirmed_not_person | skipped
    labeled_at_utc: str | None = None
    # The pipeline decision when this row entered the labeling queue. Normal
    # active learning queues only REVIEW rows; a balanced audit also samples
    # ACCEPTED and classifier-ready REJECTED rows. Keeping the original
    # bucket makes the training data auditable and lets a reviewer check that
    # each labeling batch has useful positive and negative coverage.
    original_decision: str = "review"

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
            "context_text": self.context_text,
            "status": self.status,
            "labeled_at_utc": self.labeled_at_utc,
            "original_decision": self.original_decision,
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
        self._existing_texts_cache: set[str] | None = None

    def log_review_persons(self, persons: list[AggregatedPerson], source_file: str, document_text: str) -> int:
        """Append every REVIEW-bucket person's first mention to the
        pending feedback log. Deduplicates against text already pending
        or already confirmed, so re-running on the same file doesn't
        pile up duplicate labeling work."""
        return self.append_new(self.build_records(persons, source_file, document_text))

    @staticmethod
    def build_records(persons: list[AggregatedPerson], source_file: str, document_text: str) -> list[FeedbackRecord]:
        """Build (but don't write or dedupe) one record per REVIEW person.
        Split out from log_review_persons() so parallel scan workers can
        build records next to the document text they need, then hand them
        to the MAIN process for the single, deduplicated append_new() -
        see scripts/scan_directory.py. Workers writing directly each held
        their own dedupe cache, so the same name could be logged once per
        worker and concurrent appends could interleave lines."""
        records: list[FeedbackRecord] = []
        for person in persons:
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
                context_text=line_containing(document_text, first_mention.candidate.start),
            )
            records.append(record)
        return records

    @staticmethod
    def build_candidate_records(
        candidates: list[CandidateResult], source_file: str, document_text: str
    ) -> list[FeedbackRecord]:
        """Build label-ready records from individual pipeline candidates.

        This is used by the stratified real-data sampler. Unlike the normal
        REVIEW queue it can include accepted and rejected candidates, but it
        deliberately skips hard-rejected rows with no feature vector: those
        never reach LightGBM during live inference and cannot improve that
        model's training data.
        """
        records: list[FeedbackRecord] = []
        for candidate in candidates:
            feature_vector = candidate.state.feature_vector
            if feature_vector is None:
                continue
            decision = candidate.state.decision
            record = FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                source_file=source_file,
                text=candidate.candidate.text,
                normalized_text=candidate.candidate.normalized_text,
                location=str(candidate.candidate.location) if candidate.candidate.location else "",
                confidence=candidate.state.final_confidence,
                features=feature_vector.as_list(),
                logged_at_utc=datetime.now(timezone.utc).isoformat(),
                context_text=line_containing(document_text, candidate.candidate.start),
                original_decision=decision.value if decision is not None else "unknown",
            )
            records.append(record)
        return records

    def append_new(self, records: list[FeedbackRecord]) -> int:
        """Append records whose text isn't already pending or confirmed
        (or earlier in this same list). Returns how many were written."""
        already_seen = self._existing_texts()

        new_records: list[FeedbackRecord] = []
        for record in records:
            key = record.normalized_text.lower()
            if key in already_seen:
                continue
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
        for line in _read_jsonl_lines(self.pending_path):
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
        for line in _read_jsonl_lines(self.confirmed_path):
            records.append(FeedbackRecord.from_dict(json.loads(line)))
        return records

    def _existing_texts(self) -> set[str]:
        """Cached after the first call (2026-09-22 profiling finding:
        this previously re-read AND re-parsed both full JSONL files from
        disk on every single log_review_persons() call - i.e. once per
        document in a batch scan. With confirmed_labels.jsonl at 1,300+
        lines and growing, that was measured at ~1.6-1.8s per file,
        larger than every other pipeline stage combined - for a stage
        that should just be appending a few new lines. A FeedbackStore
        instance lives for the whole batch-scan run (one Pipeline, one
        PersonExtractor, one FeedbackStore - see person_extractor.py's
        __init__), and log_review_persons() already mutates the returned
        set in place as it logs new records, so caching the set itself
        (not a copy) keeps it correctly up to date across calls within
        that one run with zero extra bookkeeping. Not safe to assume
        stale-free across SEPARATE processes sharing the same files
        (e.g. label_feedback.py confirming records while a scan is
        mid-run in another process) - not a real scenario for how this
        project is actually used (batch scans and labeling sessions are
        run as separate, sequential invocations), so not guarded against."""
        if self._existing_texts_cache is None:
            seen = set()
            for r in self.load_pending():
                seen.add(r.normalized_text.lower())
            for r in self.load_confirmed():
                seen.add(r.normalized_text.lower())
            self._existing_texts_cache = seen
        return self._existing_texts_cache
