"""
src.io.docx_reader
====================
Reader for Word .docx, standard library only (a .docx is a zip of XML
parts - no python-docx dependency, keeping the project fully offline and
lean). Each PARAGRAPH becomes exactly one line (an in-paragraph line break
becomes a space), so "line N" in a result is paragraph N of the body.

Read, in order: the body (paragraphs and table cells, in document
order), then footnotes, endnotes and comments, then the document's own
author metadata ("Document author: ..." / "Last modified by: ...") - in
forensic review, who wrote or last edited a document is often exactly
the name that matters, and comments carry their author's name too.
Legacy binary .doc files are a different format and are not supported.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.markup_text import CELL_SEPARATOR

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DC = "{http://purl.org/dc/elements/1.1/}"
_CP = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"
_EXTRA_PARTS = ("word/footnotes.xml", "word/endnotes.xml", "word/comments.xml")


class DocxReader(BaseReader):
    supported_extensions = ("docx",)

    def _read(self, path: str) -> Document:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            lines = _paragraph_lines(ET.fromstring(archive.read("word/document.xml")))
            for part in _EXTRA_PARTS:
                if part in names:
                    root = ET.fromstring(archive.read(part))
                    extra = _comment_author_lines(root) + _paragraph_lines(root)
                    if extra:
                        lines += [""] + extra
            if "docProps/core.xml" in names:
                meta = _author_lines(ET.fromstring(archive.read("docProps/core.xml")))
                if meta:
                    lines += [""] + meta
        full_text = "\n".join(lines)
        return Document(
            source_path=path, source_format=SourceFormat.DOCX,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"location_note": "line N = paragraph N of the body"},
        )


def _paragraph_text(paragraph: ET.Element) -> str:
    parts = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append(" ")
    return "".join(parts).strip()


def _paragraph_lines(root: ET.Element) -> list[str]:
    """Paragraphs in document order; a table row's cells are joined on one
    line with CELL_SEPARATOR so adjacent cells can't merge into one name."""
    lines: list[str] = []
    body = root.find(f"{_W}body")
    for block in (body if body is not None else root):
        if block.tag == f"{_W}tbl":
            for row in block.iter(f"{_W}tr"):
                cells = [" ".join(_paragraph_text(p) for p in cell.iter(f"{_W}p")).strip()
                         for cell in row.findall(f"{_W}tc")]
                lines.append(CELL_SEPARATOR.join(cells))
        else:
            lines.extend(_paragraph_text(p) for p in block.iter(f"{_W}p"))
    return lines


def _comment_author_lines(root: ET.Element) -> list[str]:
    authors = {c.get(f"{_W}author") for c in root.iter(f"{_W}comment")} - {None, ""}
    return [f"Comment author: {a}" for a in sorted(authors)]


def _author_lines(core: ET.Element) -> list[str]:
    out = []
    for tag, label in ((f"{_DC}creator", "Document author"), (f"{_CP}lastModifiedBy", "Last modified by")):
        node = core.find(tag)
        if node is not None and (node.text or "").strip():
            out.append(f"{label}: {node.text.strip()}")
    return out
