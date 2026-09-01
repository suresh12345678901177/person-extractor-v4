"""
src.preprocessing.cleaner
===========================
Text normalization applied before detection: whitespace collapsing,
control-character stripping, unicode normalization. Deliberately
conservative - case, punctuation, and line breaks are preserved because
downstream detectors and the line-indexer both depend on them.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+(?=\n)")


def clean_text(text: str) -> str:
    """Normalize unicode and strip control chars WITHOUT changing the
    text's length-affecting structure where possible. Kept intentionally
    light-touch: aggressive whitespace collapsing would shift character
    offsets and break the line/column indexer's accuracy.

    Deliberately does NOT collapse runs of 2+ spaces/tabs mid-line (an
    earlier version did, via a since-removed _MULTI_SPACE_RE pass) - that
    directly contradicted this function's own stated intent above. Multi-
    space runs are a real, meaningful structural signal in tabular/report-
    style text (column padding: "Name        Department   Status"), and
    regex_detector.py's _SAFE_SEP already treats 3+ consecutive spaces as
    a hard boundary a name can't cross for exactly this reason - but that
    logic is powerless if the multi-space run has already been silently
    collapsed to a single space before detection ever runs. Trailing
    whitespace right before a newline is still trimmed (harmless: never
    part of a name, and only shifts offsets after the line's own content).
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_RE.sub("", text)
    return text
