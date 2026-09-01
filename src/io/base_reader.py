"""
src.io.base_reader
====================
Base interface every reader implements. Readers ONLY read: detect file
type, load the file, and produce a DocumentResult. They must never
perform any extraction/NLP logic (single-responsibility, per the
blueprint's Design Principles).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.models import Document, DocumentResult
from src.utils.logger import get_logger

logger = get_logger("io.base_reader")


class BaseReader(ABC):
    """Every concrete reader implements `_read(path) -> Document`."""

    #: file extensions (without dot, lowercase) this reader supports
    supported_extensions: tuple[str, ...] = ()

    def supports(self, path: str | Path) -> bool:
        ext = Path(path).suffix.lower().lstrip(".")
        return ext in self.supported_extensions

    def read(self, path: str | Path) -> DocumentResult:
        """Never raises. Wraps _read() and converts exceptions into a
        structured, failed DocumentResult - no silent failure."""
        path = str(path)
        if not Path(path).exists():
            msg = f"File not found: {path}"
            logger.error(msg)
            return DocumentResult(success=False, source_path=path, error=msg)

        try:
            document = self._read(path)
            return DocumentResult(success=True, document=document, source_path=path)
        except Exception as exc:  # noqa: BLE001 - deliberate: never crash the pipeline
            logger.exception("Failed to read %s", path)
            return DocumentResult(success=False, source_path=path, error=str(exc))

    @abstractmethod
    def _read(self, path: str) -> Document:
        """Concrete implementation: parse the file and return a Document.
        May raise; the base class converts exceptions into DocumentResult."""
        raise NotImplementedError
