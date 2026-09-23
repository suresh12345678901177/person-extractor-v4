"""
src.io.pdf_reader
====================
Reader for .pdf files. Extracts the embedded text layer only - a
scanned/image-only PDF with no text layer yields an empty string per
page, same as a corrupted or unreadable file would; it is never sent
through OCR here (see language_filter's MIN_WORDS_FOR_LANGUAGE_CHECK
guard, which keeps a near-empty document from being misjudged either
way).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader as _PypdfReader

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader

# pypdf logs a WARNING per malformed/unusual font it can't fully parse
# (e.g. "fontTools is required to fully parse...") - harmless for text
# extraction (it falls back to a partial mapping) but extremely noisy
# on real-world PDFs with embedded custom fonts, at real-world case
# volume. Text extraction quality is unaffected; only log verbosity is.
logging.getLogger("pypdf").setLevel(logging.ERROR)


class PdfReader(BaseReader):
    supported_extensions = ("pdf",)

    def _read(self, path: str) -> Document:
        reader = _PypdfReader(path)
        pages = tuple(
            Page(index=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        )
        full_text = "\n\n".join(p.text for p in pages)
        return Document(
            source_path=path,
            source_format=SourceFormat.PDF,
            pages=pages,
            full_text=full_text,
            metadata={"page_count": len(pages)},
        )
