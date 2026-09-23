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


def line_containing(text: str, offset: int, max_chars: int = 2000) -> str:
    """Return the line containing offset, bounded to at most max_chars -
    used for context-window feature extraction and
    FeedbackRecord.context_text capture.

    max_chars exists because a "line" is only as bounded as the nearest
    newline: previously this had no cap at all, so a document (or
    document region) with no newline anywhere near offset - a minified
    JS bundle, a single-paragraph book dump, a chat export whose "line"
    spans the entire conversation - returned the ENTIRE remaining
    document. Found via a real, git-tracked-file audit (2026-09-23):
    this produced up to 927,834-char single-record captures in
    practice, including one real case's full chat conversation (both
    parties' phone numbers included) for what should have been a short
    local snippet around 3 unrelated non-person tokens. When the raw
    line exceeds max_chars, the window is re-centered on offset (not
    just truncated from window_start) so the candidate itself is never
    cut out of its own "context"."""
    window_start = text.rfind("\n", 0, offset)
    window_start = 0 if window_start == -1 else window_start + 1
    window_end = text.find("\n", offset)
    window_end = len(text) if window_end == -1 else window_end

    if window_end - window_start > max_chars:
        half = max_chars // 2
        window_start = max(window_start, offset - half)
        window_end = min(window_end, offset + half)

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


_PRECEDING_WORD_RE = re.compile(r"([A-Za-z']+)\s*$")
_LOOKAROUND_INITIAL_WINDOW = 100


def preceding_word(text: str, start: int) -> str:
    """Return the word immediately preceding a span's start offset, or
    empty string if the span starts the line.

    Uses a bounded lookback window rather than text[:start] in full -
    called per-candidate from both feature extraction and
    ContextValidator, so on a large document with many candidates a
    full-prefix copy PLUS an unanchored regex search over it per call is
    O(candidates x document_length) overall, and can be far worse than
    that: re.search with no '^' anchor and no match near the string's
    end (extremely common in dense structured text - log lines full of
    brackets/digits/identifiers) must probe many starting positions
    before concluding "no match," which is its own severe cost
    independent of the slice copy. Confirmed directly: the dominant cost
    behind a multi-hour hang on a real 13.7MB, 53,587-candidate case
    file traced to exactly this call.

    Widens the window only when the match actually touches the window's
    left edge (the word might continue further left) rather than
    guessing from a single boundary character - a guess-based version
    of this fix was tried and rejected: on dense alphanumeric log text,
    a boundary character is "inside a word" often enough that it fell
    back to the full, catastrophically slow regex almost every call,
    barely improving anything."""
    window = _LOOKAROUND_INITIAL_WINDOW
    while True:
        window_start = max(0, start - window)
        before = text[window_start:start].rstrip()
        match = _PRECEDING_WORD_RE.search(before)
        if not match:
            return ""
        if match.start(1) > 0 or window_start == 0:
            return match.group(1)  # word doesn't touch the window edge - complete
        window *= 4  # word touches the edge - it may continue further left; widen and retry


_CUE_WORD_RE = re.compile(r"[A-Za-z']+")


def preceding_words(text: str, start: int, n: int = 3) -> list[str]:
    """Return up to n words immediately before start, nearest-first,
    bounded to the candidate's own line. Used for the person/non-person
    context-cue features (see feature_extractor.py) - a separate, wider
    helper from preceding_word() above (which returns only the single
    immediate word and is relied on elsewhere for stopword-adjacency
    checks); kept as its own function rather than generalizing that one
    so existing callers' exact behavior can't shift under them.
    Deliberately does not cross into a previous line - a word from an
    unrelated prior log/chat line is not real context for this candidate."""
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start == -1 else line_start + 1
    before = text[line_start:start]
    words = _CUE_WORD_RE.findall(before)
    return list(reversed(words[-n:])) if words else []


def following_words(text: str, end: int, n: int = 3) -> list[str]:
    """Return up to n words immediately after end, nearest-first,
    bounded to the candidate's own line. Deliberately stops at the line
    boundary rather than reading into the next line - see
    following_char()'s docstring above: a name sitting alone on its own
    chat-export line, with nothing following it on that line, is real,
    load-bearing signal for this project's actual casework, and a
    previous attempt at reaching past a newline here for a different
    feature measurably collapsed accepted-only recall (31.55% -> 8.65%,
    one benchmark file's recall to 0.00%). This helper simply returns no
    words in that case rather than ever repeating that mistake."""
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    after = text[end:line_end]
    words = _CUE_WORD_RE.findall(after)
    return words[:n]


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
    window = _LOOKAROUND_INITIAL_WINDOW
    while True:
        window_end = min(len(text), end + window)
        rest = text[end:window_end].lstrip(" \t")
        if rest:
            return rest[0]
        if window_end == len(text):
            return ""
        window *= 4  # entire window was spaces/tabs - widen and retry
