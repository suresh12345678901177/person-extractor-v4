"""
src.io.xml_reader
===================
Reader for .xml (Android shared_prefs/contacts exports, app data, office
XML). Every element's text becomes a line, and so does every attribute
value that contains a letter, as "attribute: value" - names often live in
attributes (<contact name="Andre Silva" .../>), not element text.

Malformed XML (common in carved/partial forensic exports) falls back to
the lenient HTML text extractor instead of failing, so its text is still
read. ElementTree does not resolve external entities, so parsing an
untrusted export can't reach outside the file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.binary_check import binary_document, binary_format
from src.io.markup_text import html_to_text
from src.io.text_reader import TextReader
from src.utils.logger import get_logger

logger = get_logger("io.xml_reader")


class XmlReader(BaseReader):
    supported_extensions = ("xml",)

    def _read(self, path: str) -> Document:
        raw = Path(path).read_bytes()
        binary = binary_format(raw)
        if binary:
            return binary_document(path, SourceFormat.XML, binary)
        try:
            full_text = "\n".join(xml_to_lines(ET.fromstring(raw)))
        except ET.ParseError as exc:
            logger.warning("%s is not well-formed XML (%s) - extracting its text leniently", path, exc)
            full_text = html_to_text(TextReader._decode(raw))
        return Document(
            source_path=path, source_format=SourceFormat.XML,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"location_note": "line N of the extracted text/attribute lines"},
        )


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]  # drop "{namespace}"


def xml_to_lines(root: ET.Element) -> list[str]:
    lines: list[str] = []

    def add(text: str | None, label: str = "") -> None:
        text = " ".join((text or "").split())
        if text and any(ch.isalpha() for ch in text):
            lines.append(f"{label}: {text}" if label else text)

    for element in root.iter():
        for attr, value in element.attrib.items():
            add(value, _local(attr))
        add(element.text)
        add(element.tail)
    return lines
