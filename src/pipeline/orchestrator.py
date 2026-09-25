"""
src.pipeline.orchestrator
============================
The Pipeline class orchestrates the stages every extraction type
shares - reading the input file and cleaning its text - then hands off
to a per-extraction-type `Extractor` (src.extraction.base_extractor)
for everything specific to what's actually being pulled out of that
text. It deliberately knows nothing about persons, phone numbers, IDs,
or any other entity kind; that knowledge lives entirely inside each
Extractor implementation (src.extraction.person_extractor.PersonExtractor
today).

Every stage is wrapped in `timed_stage`, so the final ExtractionResult
carries a complete, labeled timing breakdown - the "time taken for the
task" output requirement. Failures in one document never crash the
whole run - they produce a structured, failed ExtractionResult instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.document_factory import DocumentFactory
from src.core.models import ExtractionResult, PipelineStatistics, SourceFormat
from src.extraction.registry import get_extractor
from src.preprocessing.cleaner import clean_text
from src.preprocessing.line_indexer import LineIndex
from src.utils.logger import get_logger
from src.utils.timing import Stopwatch, timed_stage

logger = get_logger("pipeline.orchestrator")

# Same separator PdfReader joins pages with (src/io/pdf_reader.py).
_PAGE_SEPARATOR = "\n\n"


class Pipeline:
    def __init__(self, config: dict[str, Any], base_dir: str | Path) -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.document_factory = DocumentFactory()

        # Single-extractor-per-run for now (see config.py's "extraction"
        # block docstring) - only the first configured type runs. Running
        # several extractors in one pass and merging/labeling their
        # results is a real design question, deliberately deferred to
        # whenever a second extraction type actually exists to design it
        # against.
        extractor_name = config.get("extraction", {}).get("enabled_types", ["person"])[0]
        self.extractor = get_extractor(extractor_name)(config, self.base_dir)

    def run(self, input_path: str | Path) -> ExtractionResult:
        input_path = str(input_path)
        stats = PipelineStatistics()
        total_timer = Stopwatch()

        with timed_stage("io_read_document", stats.stage_timings):
            doc_result = self.document_factory.create(input_path)

        if not doc_result.success or doc_result.document is None:
            stats.documents_failed += 1
            stats.total_seconds = total_timer.elapsed
            return ExtractionResult(
                success=False, source_path=input_path,
                error=doc_result.error, statistics=stats,
            )

        stats.documents_processed += 1
        document = doc_result.document
        if document.metadata.get("skipped_reason") == "binary_content":
            # A text extension with binary content (src/io/binary_check.py):
            # reported as skipped, never run through the detectors.
            stats.total_seconds = total_timer.elapsed
            return ExtractionResult(
                success=True, source_path=input_path, statistics=stats,
                model_info={"skipped_reason": "binary_content",
                            "binary_format": document.metadata.get("binary_format", "")},
            )
        pages = [p.text for p in document.pages] if document.source_format == SourceFormat.PDF else None
        return self._run_on_text(
            document.full_text, input_path, stats, total_timer, pages=pages, source_format=document.source_format,
        )

    def run_text(self, text: str, source_label: str = "<text>") -> ExtractionResult:
        """Same pipeline as run(), on an in-memory string instead of a
        file - the entry point for the real-time service (server.py),
        where text arrives over HTTP and never touches disk."""
        stats = PipelineStatistics()
        stats.documents_processed += 1
        return self._run_on_text(text, source_label, stats, Stopwatch())

    def _run_on_text(
        self, raw_text: str, input_path: str, stats: PipelineStatistics, total_timer: Stopwatch,
        pages: list[str] | None = None,
        source_format: SourceFormat | None = None,
    ) -> ExtractionResult:
        with timed_stage("preprocessing", stats.stage_timings):
            if pages is None:
                cleaned_text = clean_text(raw_text)
                line_index = LineIndex.build(cleaned_text)
            else:
                # PDFs: still ONE document for extraction (repetition and
                # name propagation should see every page), but each page is
                # cleaned on its own and joined with the same blank line
                # PdfReader uses, so every page's start offset in the
                # cleaned text is known exactly and locations can say
                # "page 7, line 12" instead of a line of the joined text.
                cleaned_pages = [clean_text(p) for p in pages]
                page_starts, offset = [], 0
                for page in cleaned_pages:
                    page_starts.append(offset)
                    offset += len(page) + len(_PAGE_SEPARATOR)
                cleaned_text = _PAGE_SEPARATOR.join(cleaned_pages)
                line_index = LineIndex.build(cleaned_text, page_start_offsets=page_starts)

        result = self.extractor.extract(cleaned_text, line_index, input_path, stats, source_format=source_format)
        stats.total_seconds = total_timer.elapsed

        logger.info(
            "Pipeline finished for %s in %.3fs (total, including io/preprocessing)",
            input_path, stats.total_seconds,
        )

        return result
