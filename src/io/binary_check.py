"""
src.io.binary_check
======================
Detects files that carry a text extension but binary content, so the
converting readers skip them instead of decoding bytes into "words".

Found on a real case scan (2026-09-24): 301 of 2,517 .xml files in an
Android extraction were binary - Android Binary XML ("ABX\\0", the format
Android 12+ uses for many system files) and compiled resource XML
(AndroidManifest.xml inside APKs). XML parsing failed, the lenient
fallback decoded the raw bytes, and fragments came out as accepted
"names" like "ÑËÙ".
"""

from __future__ import annotations

_KNOWN_FORMATS = (
    (b"ABX\x00", "Android binary XML (ABX)"),
    (b"\x03\x00\x08\x00", "compiled Android resource XML"),
)
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_SAMPLE_BYTES = 8192


def binary_format(raw: bytes) -> str | None:
    """A short description if `raw` is binary content, else None. UTF-16
    text (BOM-marked; it legitimately contains NUL bytes) is text."""
    for magic, name in _KNOWN_FORMATS:
        if raw.startswith(magic):
            return name
    if raw.startswith(_UTF16_BOMS):
        return None
    sample = raw[:_SAMPLE_BYTES]
    if not sample:
        return None
    if b"\x00" in sample:
        return "binary data (NUL bytes)"
    control = sum(1 for b in sample if b < 9 or 13 < b < 32)
    if control / len(sample) > 0.10:
        return "binary data (control bytes)"
    return None


def binary_document(path: str, source_format, description: str):
    """An empty Document flagged so the pipeline reports the file as
    skipped (binary content) rather than as read-with-no-names."""
    from src.core.models import Document, Page

    return Document(
        source_path=path, source_format=source_format,
        pages=(Page(index=0, text=""),), full_text="",
        metadata={"skipped_reason": "binary_content", "binary_format": description},
    )
