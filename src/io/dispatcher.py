"""
src.io.dispatcher
===================
Detects a file's type by extension and dispatches to the appropriate
reader. The open/closed registry pattern means adding a future reader
(e.g. a DocxReader) is a one-line addition here with zero changes to
any other file - this is the "Plugin Architecture" design principle
applied to the IO layer.
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import DocumentResult
from src.io.base_reader import BaseReader
from src.io.csv_reader import CsvReader
from src.io.docx_reader import DocxReader
from src.io.email_reader import EmailReader
from src.io.html_reader import HtmlReader
from src.io.json_reader import JsonReader
from src.io.pdf_reader import PdfReader
from src.io.text_reader import TextReader
from src.io.xml_reader import XmlReader
from src.utils.logger import get_logger

logger = get_logger("io.dispatcher")


class ReaderDispatcher:
    """Registry of readers, selected by supported file extension."""

    def __init__(self) -> None:
        self._readers: list[BaseReader] = [
            TextReader(),
            PdfReader(),
            # Standard-library converters (2026-09-24) - no new dependencies.
            CsvReader(),
            JsonReader(),
            HtmlReader(),
            XmlReader(),
            DocxReader(),
            EmailReader(),
            # Future formats register here, e.g.:
            #   ExcelReader(), ImageReader()
            # No other file needs to change when a new reader is added.
        ]

    def supported_extensions(self) -> tuple[str, ...]:
        """Every extension some reader handles, lowercase with a leading
        dot (".txt", ".csv", ...) - the single list scan_directory.py and
        error messages use, so a new reader is picked up everywhere."""
        return tuple(f".{ext}" for reader in self._readers for ext in reader.supported_extensions)

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
