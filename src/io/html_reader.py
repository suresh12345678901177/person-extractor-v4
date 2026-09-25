"""
src.io.html_reader
====================
Reader for .html / .htm (saved web pages, webmail and chat exports).
Only visible text is kept - scripts, styles and similar are dropped - and
block elements / table cells keep their boundaries (see markup_text.py).
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.binary_check import binary_document, binary_format
from src.io.markup_text import html_to_text
from src.io.text_reader import TextReader


class HtmlReader(BaseReader):
    supported_extensions = ("html", "htm", "xhtml")

    def _read(self, path: str) -> Document:
        raw = Path(path).read_bytes()
        binary = binary_format(raw)
        if binary:
            return binary_document(path, SourceFormat.HTML, binary)
        full_text = html_to_text(TextReader._decode(raw))
        return Document(
            source_path=path, source_format=SourceFormat.HTML,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"location_note": "line N of the extracted visible text"},
        )
