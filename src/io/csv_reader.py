"""
src.io.csv_reader
===================
Reader for .csv / .tsv (contact lists, call logs, message tables - common
in forensic exports). Each RECORD becomes exactly one line, cells joined
by CELL_SEPARATOR, so "line N" in a result is record N (the header, if
any, is line 1) and no name can run from one cell into the next. Line
breaks inside a quoted cell are flattened to spaces to keep that
one-record-one-line guarantee.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.binary_check import binary_document, binary_format
from src.io.markup_text import CELL_SEPARATOR
from src.io.text_reader import TextReader

# Real exports have multi-MB cells (embedded message bodies, base64);
# the csv module's 128KB default would raise on them.
csv.field_size_limit(2**31 - 1)


class CsvReader(BaseReader):
    supported_extensions = ("csv", "tsv")

    def _read(self, path: str) -> Document:
        raw = Path(path).read_bytes()
        binary = binary_format(raw)
        if binary:
            return binary_document(path, SourceFormat.CSV, binary)
        text = TextReader._decode(raw)
        delimiter = "\t" if path.lower().endswith(".tsv") else _sniff_delimiter(text)
        lines = [
            CELL_SEPARATOR.join(" ".join(cell.split()) for cell in row)
            for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        ]
        full_text = "\n".join(lines)
        return Document(
            source_path=path, source_format=SourceFormat.CSV,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"record_count": len(lines), "delimiter": delimiter,
                      "location_note": "line N = CSV record N"},
        )


def _sniff_delimiter(text: str) -> str:
    """The delimiter that occurs most in the first non-empty line (the
    header, in real exports). Deliberately not csv.Sniffer: it misread a
    semicolon file (European Excel's default) as comma-separated when a
    quoted cell spanned lines - the whole file then became one column."""
    first = next((line for line in text.split("\n") if line.strip()), "")
    counts = {d: first.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","
