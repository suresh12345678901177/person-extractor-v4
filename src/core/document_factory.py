"""
src.core.document_factory
===========================
DocumentFactory is the single entry point the pipeline uses to go from
a file path to a ready-to-process Document: it wraps the IO dispatcher
and attaches document-level metadata (file size, line count, read
timestamp) that later stages and the final report depend on.

Kept in `core` (rather than `io`) because it produces the `Document`
core model and is the natural extension point if document-level
validation (e.g. "reject empty files", "reject files over N MB") is
ever needed - that policy belongs next to the model it protects, not
buried inside a specific reader.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from src.core.models import DocumentResult
from src.io.dispatcher import ReaderDispatcher
from src.utils.logger import get_logger

logger = get_logger("core.document_factory")


class DocumentFactory:
    def __init__(self, dispatcher: ReaderDispatcher | None = None) -> None:
        self._dispatcher = dispatcher or ReaderDispatcher()

    def create(self, path: str | Path) -> DocumentResult:
        result = self._dispatcher.dispatch(path)
        if not result.success or result.document is None:
            return result

        file_path = Path(path)
        enriched_metadata = dict(result.document.metadata)
        enriched_metadata.update({
            "file_size_bytes": file_path.stat().st_size,
            "line_count": result.document.full_text.count("\n") + 1,
            "char_count": len(result.document.full_text),
            "read_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })

        document = result.document.__class__(
            source_path=result.document.source_path,
            source_format=result.document.source_format,
            pages=result.document.pages,
            full_text=result.document.full_text,
            metadata=enriched_metadata,
        )
        logger.info(
            "Document created: %s (%d bytes, %d lines, %d chars)",
            path, enriched_metadata["file_size_bytes"],
            enriched_metadata["line_count"], enriched_metadata["char_count"],
        )
        return DocumentResult(success=True, document=document, source_path=str(path))
