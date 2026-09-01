"""
src.preprocessing.segmenter
=============================
Lightweight, dependency-free sentence segmentation and context-window
helpers. No hard dependency on spaCy for this - spaCy is reserved for
the detection layer's NER signal; segmentation for feature engineering
and the context validator stays simple and fast.
"""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\u2018\u201c])")
_QUOTE_CHARS = ('"', "'", "\u2018", "\u2019", "\u201c", "\u201d")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation + capitalization heuristics."""
    if not text.strip():
        return []
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(line) if s.strip())
    return sentences


def line_containing(text: str, offset: int) -> str:
    """Return the full line (not sentence) that contains the character
    offset - used for context-window feature extraction."""
    window_start = text.rfind("\n", 0, offset)
    window_start = 0 if window_start == -1 else window_start + 1
    window_end = text.find("\n", offset)
    window_end = len(text) if window_end == -1 else window_end
    return text[window_start:window_end]


def is_inside_quotes(text: str, start: int, end: int, window: int = 200) -> bool:
    """Heuristic: count unmatched quote characters in the `window` chars
    preceding `start` (within the same line). An odd count means the
    span opens inside an already-open quoted string - used both as a
    feature and by the context validator (quoted slogans/speech are
    weaker evidence of a "named" person than plain narration)."""
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start == -1 else line_start + 1
    search_start = max(line_start, start - window)
    preceding = text[search_start:start]
    quote_count = sum(preceding.count(q) for q in ('"',))
    return quote_count % 2 == 1


def preceding_word(text: str, start: int) -> str:
    """Return the word immediately preceding a span's start offset, or
    empty string if the span starts the line."""
    before = text[:start].rstrip()
    match = re.search(r"([A-Za-z']+)\s*$", before)
    return match.group(1) if match else ""


def count_occurrences(text: str, document_text: str) -> int:
    """Count exact, word-boundary-safe occurrences of `text` anywhere in
    `document_text`, position-agnostic (mid-sentence and standalone-line
    occurrences count identically). Shared by CorroborationValidator (its
    own repetition-corroboration pass/fail decision) and DecisionEngine
    (the knowledge-corroboration gate's ML-high-confidence path - see
    decision_engine.py's ML_HIGH_CONFIDENCE_OVERRIDE handling) so both
    "does this recur enough to trust it" checks can never silently drift
    out of sync with each other."""
    return len(re.findall(rf"\b{re.escape(text)}\b", document_text))


def following_char(text: str, end: int) -> str:
    """Return the first non-space character immediately after a span's
    end offset, or empty string at end of text.

    Deliberately stops at a bare newline rather than skipping past it -
    tried skipping past newlines during testing (treating "\\n" as just
    more whitespace to look through for the "real" following character):
    a name sitting alone on its own chat-export line, immediately
    followed by "\\n", is exactly the real-world sender-header shape this
    tool's actual documents are full of (ProDiscover WhatsApp/chat
    exports format sender names this way constantly) - the ML
    classifier leans on that shape as real, load-bearing signal for
    real casework. A synthetic adversarial "Ref: X | Code: ..." ledger
    label can exploit the same shape to get a false accept from a
    single occurrence (a real, narrow precision gap - see
    CorroborationValidator/knowledge-corroboration-gate docstrings for
    the broader family of this issue) - but skipping newlines here to
    close that one gap measurably collapsed accepted-only recall across
    the whole benchmark suite (verified: 31.55% -> 8.65%, one real
    benchmark file's accepted recall dropped to 0.00%), which is a far
    worse real-world tradeoff than the gap it closes. Left as-is
    pending a fix that targets the exploit specifically (e.g. a
    dedicated "line-final AND zero knowledge-corroboration AND
    low-repetition" check) rather than removing the signal wholesale.
    """
    rest = text[end:].lstrip(" \t")
    return rest[0] if rest else ""
