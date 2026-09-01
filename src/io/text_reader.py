"""
src.io.text_reader
====================
Reader for plain-text (.txt) files - the only format in V4's scope.
Handles common encodings defensively (a raw .txt from Windows Notepad,
Excel-exported CSV-as-txt, or a Linux tool can each use a different
encoding) rather than assuming UTF-8 and failing on real-world files.
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader


class TextReader(BaseReader):
    supported_extensions = ("txt",)

    def _read(self, path: str) -> Document:
        raw_bytes = Path(path).read_bytes()
        text = self._decode(raw_bytes)
        page = Page(index=0, text=text)
        return Document(
            source_path=path,
            source_format=SourceFormat.TXT,
            pages=(page,),
            full_text=text,
        )

    @staticmethod
    def _decode(raw_bytes: bytes) -> str:
        # Order matters: utf-8-sig strips a BOM if present (common from
        # Windows tools); utf-8 is the modern default; latin-1 never
        # raises (every byte is valid) so it is the last-resort fallback
        # that guarantees _decode always returns something usable.
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")
