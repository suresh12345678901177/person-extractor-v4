"""
src.io.dispatcher
===================
Detects a file's type by extension and dispatches to the appropriate
reader. V4 registers only TextReader (TXT-only scope), but the
open/closed registry pattern means adding a future reader (e.g. a
DocxReader) is a one-line addition here with zero changes to any other
file - this is the "Plugin Architecture" design principle applied to
the IO layer.
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import DocumentResult
from src.io.base_reader import BaseReader
from src.io.text_reader import TextReader
from src.utils.logger import get_logger

logger = get_logger("io.dispatcher")


class ReaderDispatcher:
    """Registry of readers, selected by supported file extension."""

    def __init__(self) -> None:
        self._readers: list[BaseReader] = [
            TextReader(),
            # Future formats register here, e.g.:
            #   DocxReader(), PdfReader(), ExcelReader(), ImageReader()
            # No other file needs to change when a new reader is added.
        ]

    def get_reader(self, path: str | Path) -> BaseReader | None:
        for reader in self._readers:
            if reader.supports(path):
                return reader
        return None

    def dispatch(self, path: str | Path) -> DocumentResult:
        reader = self.get_reader(path)
        if reader is None:
            ext = Path(path).suffix or "(no extension)"
            msg = (
                f"No reader registered for file extension '{ext}' ({path}). "
                f"PERSON_EXTRACTOR_V4 currently supports: "
                f"{', '.join(r for reader in self._readers for r in reader.supported_extensions)}"
            )
            logger.error(msg)
            return DocumentResult(success=False, source_path=str(path), error=msg)
        return reader.read(path)
