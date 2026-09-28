"""
src.candidate.name_propagation
=================================
Document-level first-name propagation: once a document has ACCEPTED a
multi-token person ("Katya Sokolova"), a standalone "Katya" in the same
document is evidence of a person too - the standard label-consistency
step in NER post-processing.

Why (measured 2026-09-24, after the benchmark gold was made consistent
about first-name mentions): single-token recall was 0.8177. The misses
were names outside the dictionary ("Katya", "Jaylen", "Kenji", "Devraj"):
22 mentions sat in REVIEW because spaCy was their only evidence and the
knowledge-corroboration gate wants dictionary/title/high-ML support -
while the SAME document had already accepted "Katya Sokolova" and
"Jaylen Brooks" - and 9 were never detected at all (a lone unknown word
nothing tagged).

What this does, after the normal decision pass:
  1. Collects first names of accepted multi-token persons, skipping any
     that are ordinary English words, known first-name/common-noun
     collisions ("Chase", "Grace", "Major") or calendar words - those
     stay under the normal rules, since a standalone "Chase" is as likely
     a bank as a person.
  2. Adds a `document_name` detection to REVIEW candidates whose text is
     one of those names; DecisionEngine treats it as knowledge
     corroboration and re-decides (score threshold still applies).
  3. Creates candidates for occurrences no detector found, and runs them
     through the FULL validation pipeline - every hard validator (system
     log lines, organizations, locations, grammar, ...) still applies.
Hard rejections are never overridden.
"""

from __future__ import annotations

import bisect
import re
from typing import Iterable

from src.core.models import CandidateResult, Decision, Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase

DOCUMENT_NAME_PATTERN = "document_name"
DOCUMENT_NAME_CONFIDENCE = 0.60


def propagatable_first_names(full_names: Iterable[str], knowledge_base: KnowledgeBase) -> dict[str, str]:
    """{first name as written: one accepted full name vouching for it}."""
    names: dict[str, str] = {}
    for full in full_names:
        tokens = [
            t for t in full.split()
            if not knowledge_base.is_title(t.rstrip(".")) and not knowledge_base.is_honorific(t.rstrip("."))
        ]
        if len(tokens) < 2:
            continue
        first = tokens[0]
        if len(first) < 2 or first.endswith(".") or not first[:1].isupper():
            continue
        if (knowledge_base.is_common_word(first) or knowledge_base.is_ambiguous_first_name(first)
                or knowledge_base.is_calendar_word(first) or knowledge_base.is_language_or_region_name(first)):
            continue
        names.setdefault(first, " ".join(tokens))
    return names


def document_name_detection(text: str, start: int, end: int, vouched_by: str) -> Detection:
    return Detection(
        text=text, start=start, end=end, page_index=0,
        detector=DetectorName.DICTIONARY, confidence=DOCUMENT_NAME_CONFIDENCE,
        metadata={"pattern": DOCUMENT_NAME_PATTERN, "vouched_by": vouched_by},
    )


def review_candidates_to_promote(
    candidates: list[CandidateResult], names: dict[str, str],
) -> list[tuple[CandidateResult, str]]:
    """REVIEW single-token candidates whose text is a propagatable first name."""
    by_key = {k.lower(): v for k, v in names.items()}
    out = []
    for c in candidates:
        if c.state.decision != Decision.REVIEW:
            continue
        text = c.candidate.normalized_text
        if len(text.split()) == 1 and text.lower() in by_key:
            out.append((c, by_key[text.lower()]))
    return out


def undetected_occurrences(
    document_text: str, names: dict[str, str], candidates: list[CandidateResult],
) -> list[Detection]:
    """Word-boundary, case-exact occurrences of the names that overlap no
    existing candidate at all (so nothing already judged is re-judged
    here). One combined regex pass, not one pass per name - this runs on
    multi-MB documents with hundreds of accepted names."""
    if not names:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b")
    taken = sorted((c.candidate.start, c.candidate.end) for c in candidates)
    starts = {s for s, _ in taken}
    detections = []
    for match in pattern.finditer(document_text):
        s, e = match.span()
        if _overlaps_any(s, e, taken):
            continue
        if _directly_precedes_a_candidate(document_text, e, starts):
            continue
        detections.append(document_name_detection(match.group(1), s, e, names[match.group(1)]))
    return detections


def _directly_precedes_a_candidate(document_text: str, end: int, candidate_starts: set[int]) -> bool:
    """True for the first half of a longer name another candidate already
    covers: "Ranjodh Aulakh" where only "Aulakh" was detected. Adding
    "Ranjodh" separately would split one name into two mentions (found on
    datasets/benchmark_unseen/: it turned one true positive into a TP plus
    a fragment). Same 1-2 space/tab tolerance as the detectors' own
    token joining (regex_detector.py's _SAFE_SEP)."""
    for gap in (1, 2):
        if document_text[end:end + gap].strip(" \t") == "" and "\n" not in document_text[end:end + gap] \
                and end + gap in candidate_starts:
            return True
    return False


# Candidate spans are names - far shorter than this - so a span starting
# more than this many chars before `start` can't reach it.
_MAX_CANDIDATE_SPAN = 500


def _overlaps_any(start: int, end: int, sorted_spans: list[tuple[int, int]]) -> bool:
    """sorted_spans is sorted by start. Only spans starting before `end`
    can overlap [start, end); walk back from there over the few that
    start close enough to still reach past `start`."""
    j = bisect.bisect_left(sorted_spans, (end,)) - 1
    while j >= 0:
        s, e = sorted_spans[j]
        if s + _MAX_CANDIDATE_SPAN < start:
            break
        if e > start:
            return True
        j -= 1
    return False
