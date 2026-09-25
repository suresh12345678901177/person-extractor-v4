"""Sample real PDF/TXT candidates from every pipeline decision bucket.

The normal feedback queue contains REVIEW candidates only. This tool adds a
balanced audit sample from ACCEPTED, REVIEW, and classifier-ready REJECTED
candidates, so retraining data reflects real pipeline behavior instead of
mostly synthetic templates or one failure bucket.

Usage:
    python scripts/sample_pipeline_candidates.py --input-dir "C:\\case_files" --queue
    python scripts/sample_pipeline_candidates.py --input-dir "C:\\case_files" --per-bucket 40 --queue

`--queue` appends the sample to datasets/feedback/pending_review.jsonl;
then run scripts/label_feedback.py. Without it, the JSONL export is only a
readable audit artifact and does not affect model training.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG
from src.core.models import CandidateResult, SourceFormat
from src.feedback.feedback_store import FeedbackStore
from src.pipeline.orchestrator import Pipeline
from src.preprocessing.cleaner import clean_text

SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt"})
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "feedback" / "pipeline_candidate_sample.jsonl"
PAGE_SEPARATOR = "\n\n"


def _input_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def _mentions(persons) -> list[CandidateResult]:
    return [mention for person in persons for mention in person.mentions]


def _pipeline_ready_text(document) -> str:
    """Mirror Pipeline's text preparation so candidate offsets locate context."""
    if document.source_format == SourceFormat.PDF:
        return PAGE_SEPARATOR.join(clean_text(page.text) for page in document.pages)
    return clean_text(document.full_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced PDF/TXT candidate sample for labeling")
    parser.add_argument("--input-dir", required=True, help="Folder containing PDF and TXT files")
    parser.add_argument("--per-bucket", type=int, default=25,
                        help="Maximum sampled candidates per accepted/review/rejected bucket (default: 25)")
    parser.add_argument("--seed", type=int, default=20260925, help="Deterministic sampling seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL audit export path")
    parser.add_argument("--queue", action="store_true", help="Append sampled rows to the interactive feedback queue")
    args = parser.parse_args()

    if args.per_bucket < 1:
        parser.error("--per-bucket must be at least 1")
    root = Path(args.input_dir).resolve()
    if not root.is_dir():
        parser.error(f"--input-dir is not a directory: {root}")
    paths = _input_files(root)
    if not paths:
        parser.error("No .pdf or .txt files found under --input-dir")

    config = copy.deepcopy(DEFAULT_CONFIG)
    # This command writes only its final, intentionally sampled rows when
    # --queue is present. Running the pipeline itself must not flood the
    # normal REVIEW queue with every candidate in the folder.
    config["feedback"]["enabled"] = False
    pipeline = Pipeline(config=config, base_dir=BASE_DIR)

    buckets: dict[str, list[tuple[CandidateResult, str, str]]] = {
        "accepted": [], "review": [], "rejected": [],
    }
    failed = 0
    for path in paths:
        result = pipeline.run(path)
        if not result.success:
            failed += 1
            continue
        document_text = pipeline.document_factory.create(path).document
        if document_text is None:
            failed += 1
            continue
        # Candidate offsets refer to this same cleaned/PDF-joined string.
        # It is used only to display the exact source line during labeling.
        raw_text = _pipeline_ready_text(document_text)
        for decision, candidates in (
            ("accepted", _mentions(result.persons)),
            ("review", _mentions(result.review_persons)),
            ("rejected", list(result.rejected)),
        ):
            buckets[decision].extend((candidate, str(path), raw_text) for candidate in candidates)

    rng = random.Random(args.seed)
    sampled: list = []
    for decision in ("accepted", "review", "rejected"):
        options = buckets[decision]
        rng.shuffle(options)
        selected = options[:args.per_bucket]
        for candidate, source_file, document_text in selected:
            sampled.extend(FeedbackStore.build_candidate_records([candidate], source_file, document_text))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in sampled:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    print(f"Scanned {len(paths)} PDF/TXT files ({failed} failed).")
    for decision in ("accepted", "review", "rejected"):
        print(f"  {decision:<8}: {len(buckets[decision])} candidates, sampled "
              f"{min(args.per_bucket, len(buckets[decision]))}")
    print(f"Wrote {len(sampled)} label-ready records to: {output}")

    if args.queue:
        store = FeedbackStore(BASE_DIR / "datasets" / "feedback")
        appended = store.append_new(sampled)
        print(f"Queued {appended} new records for labeling.")
        print("Next: python scripts/label_feedback.py")


if __name__ == "__main__":
    main()
