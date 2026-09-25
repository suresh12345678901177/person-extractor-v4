"""
src.io.json_reader
====================
Reader for .json and line-delimited .jsonl / .ndjson (app/chat exports,
API dumps). Every string value becomes a "key: value" line, in document
order - the key is kept because it is real context ("sender: Andre
Silva", "displayName: ..."). Numbers, booleans and nulls are skipped
(never names). A file that isn't valid JSON is read as plain text rather
than failing, so nothing in it is silently lost.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.binary_check import binary_document, binary_format
from src.io.text_reader import TextReader
from src.utils.logger import get_logger

logger = get_logger("io.json_reader")


class JsonReader(BaseReader):
    supported_extensions = ("json", "jsonl", "ndjson")

    def _read(self, path: str) -> Document:
        raw_bytes = Path(path).read_bytes()
        binary = binary_format(raw_bytes)
        if binary:
            return binary_document(path, SourceFormat.JSON, binary)
        raw = TextReader._decode(raw_bytes)
        line_delimited = path.lower().endswith((".jsonl", ".ndjson"))
        values = []
        parsed_ok = True
        if line_delimited:
            for line in raw.split("\n"):
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    values.append(line)  # keep the raw line's text
        else:
            try:
                values.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("%s is not valid JSON (%s) - reading it as plain text", path, exc)
                parsed_ok = False

        full_text = "\n".join(json_to_lines(values)) if parsed_ok else raw
        return Document(
            source_path=path, source_format=SourceFormat.JSON if parsed_ok else SourceFormat.TXT,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"location_note": "line N of the extracted 'key: value' text"},
        )


def json_to_lines(value) -> list[str]:
    """Depth-first, document-order 'key: value' lines for every string.
    Iterative (explicit stack), so deeply nested exports can't hit the
    recursion limit."""
    lines: list[str] = []
    stack = [(value, "")]
    while stack:
        item, key = stack.pop()
        if isinstance(item, dict):
            stack.extend(reversed([(v, str(k)) for k, v in item.items()]))
        elif isinstance(item, list):
            stack.extend(reversed([(v, key) for v in item]))
        elif isinstance(item, str):
            text = item.strip()
            if text:
                lines.append(f"{key}: {text}" if key else text)
    return lines
