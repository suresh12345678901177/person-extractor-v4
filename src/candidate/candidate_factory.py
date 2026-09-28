"""
src.candidate.candidate_factory
=================================
CandidateFactory turns raw Detection objects into Candidate /
CandidateResult objects:
  1. merge_detections()  - groups overlapping detections (possibly from
     different detectors) into clusters
  2. build_candidates()  - normalizes each cluster into one Candidate,
     seeds initial detector evidence, and attaches a TextLocation via
     the LineIndex

Named "Candidate Factory" per the blueprint's Pipeline Flow section.
"""

from __future__ import annotations

import re

from src.core.models import Candidate, CandidateResult, CandidateState, Detection, DetectorName
from src.detection.dictionary_detector import _strip_possessive
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.latin import LATIN_LETTERS, LATIN_UPPER
from src.preprocessing.line_indexer import LineIndex

_WHITESPACE_RE = re.compile(r"\s+")

# Boundary-trust priority used when overlapping detections disagree on
# where a name span starts/ends (lower = more trusted boundary). This is
# deliberately NOT the same ordering as detector confidence: the regex
# bare-capitalized-sequence pattern has decent confidence (0.55) as a
# "is this plausibly a name" signal, but it is purely a capitalization
# heuristic with zero grammatical understanding, so it is the LEAST
# trustworthy for exact boundaries - it has no way to know it just
# grabbed the next sentence's first word too (its own docstring already
# calls it "a weaker signal"). spaCy's NER is grammar/sentence-aware, so
# even though its raw confidence (0.50) is lower than bare regex's, its
# boundaries are more trustworthy and should win when they disagree.
# Titled patterns ("Dr. Suresh Kumar") are explicit and unambiguous, so
# they always win.
#
# "dictionary" (a multi-token window where every token is an
# independently-confirmed known first/last name) was missing from this
# list entirely until found via real testing (case 930204725) - it fell
# through to the unranked default (worse than even "bare"), so
# DictionaryDetector's window match ALWAYS lost boundary selection to
# both spaCy and bare regex, even when it was the objectively correct
# span. Concretely: "Baljeet Singh Sandhu" - DictionaryDetector's window
# extension correctly found the full three-token name (all three tokens
# are known names), but spaCy only tagged "Singh Sandhu" (missed the
# first name) and, because "dictionary" wasn't ranked, spaCy's narrower,
# wrong span won anyway - truncating a 94-occurrence real name down to a
# 13-occurrence fragment. A dictionary window match is explicit,
# non-statistical confirmation (every token independently verified
# against the knowledge base), so it's ranked ABOVE spaCy's statistical
# guess, just below the unambiguous "titled" pattern.
# "dictionary_single_token" (a lone token matching one known name, no
# window) is weaker - only one token confirmed, not a pair - so it's
# ranked alongside spaCy rather than above it; the existing width
# tie-break (see _select_canonical_span) still lets a wider spaCy span
# win over a single dictionary token when they're both eligible.
_PATTERN_BOUNDARY_PRIORITY = {
    "titled": 0,
    "dictionary": 1,
    "spacy_ner": 2,
    "dictionary_single_token": 2,
    "bare": 3,
}


def _boundary_priority(detection: Detection) -> int:
    return _PATTERN_BOUNDARY_PRIORITY.get(detection.metadata.get("pattern", ""), 3)


def _select_canonical_span(group: list[Detection], knowledge_base: KnowledgeBase | None = None) -> Detection:
    """Picks the most boundary-trustworthy detection's span (see
    _PATTERN_BOUNDARY_PRIORITY above) - EXCEPT one narrow, specifically
    verified rescue: when spaCy tagged exactly ONE token as the person,
    and a "bare" regex detection in the same cluster fully contains that
    token in an exactly-TWO-token span, prefer the wider bare span.

    Found via real testing (case 930204710): "Renata Kowalski" - bare
    regex correctly caught the full two-token name, but spaCy tagged
    only "Kowalski" (missed the first name, a known spaCy weakness on
    less-common names). Trusting spaCy's narrower span by default
    truncated the candidate to just "Kowalski", losing this specific
    mention's contribution to "Renata Kowalski"'s count entirely.

    Deliberately NOT a general "prefer wider bare span" rule - an
    earlier, broader version of this exact fix was tried and reverted
    (see project history) for also widening "Andre Silva" (spaCy, 2
    correct tokens) to "Andre Silva Hi I" (bare regex, 4 tokens - greedily
    matched into the next chat message). The token-count gate here
    structurally cannot fire on that case (spaCy's span there is 2
    tokens, not 1), so it only ever rescues the specific "spaCy
    recognized the person but missed exactly one adjacent name token"
    shape, never a multi-token over-match.
    """
    ranked = sorted(group, key=lambda d: (_boundary_priority(d), -(d.end - d.start)))
    best = ranked[0]

    best_pattern = best.metadata.get("pattern")
    if best_pattern in ("spacy_ner", "dictionary_single_token") and len(best.text.split()) == 1:
        for d in group:
            if d.metadata.get("pattern") != "bare":
                continue
            # Two tokens, or a middle-initial name ("Kurt D. DelBene",
            # 2026-09-28): the 3-token shape was never widened, so the lone
            # dictionary hit "Kurt" won and the surname was lost.
            if not (d.start <= best.start and d.end >= best.end
                    and (len(d.text.split()) == 2 or is_middle_initial_name(d.text))):
                continue
            if best_pattern == "spacy_ner":
                return d
            # dictionary_single_token (added 2026-09-24): the same rescue,
            # but only when the OTHER token itself reads as a name - a
            # dictionary hit, unlike spaCy, has no sentence context, so
            # without this check "Thanks Nadia" or "Nadia Report" would
            # widen too. Measured: 10.5% of accepted benchmark mentions
            # were truncated to one token, most of them exactly this shape
            # ("Nadia" of "Nadia Kovalenko", "Doran" of "Doran Vestring",
            # "Bhatt" of "Devraj Bhatt") - a known first/last name next to
            # a surname/first name no list contains.
            # Deliberately NOT also refusing when the dictionary word itself
            # is ambiguous ("Wild", "Marcus"): measured 2026-09-24, that
            # guard cost more than it saved - widening "Wild" to "Wild
            # Gateway" (a "Ref:" ledger label) lets the validators reject
            # the whole phrase, where "Wild" alone was accepted (+10 false
            # positives with the guard, vs 2 "Marcus Nadka" mentions saved).
            other = d.text.split()[-1] if d.start == best.start else d.text.split()[0]
            if knowledge_base is not None and _is_plausible_name_token(other, knowledge_base):
                return d
    return best


_NAME_WORD_RE = re.compile(rf"^[{LATIN_UPPER}][{LATIN_LETTERS}'\-]+$")
_INITIAL_RE = re.compile(rf"^[{LATIN_UPPER}]\.$")


def is_middle_initial_name(text: str) -> bool:
    """True for 'Craig J. Mundie' / 'Mary A. B. Jones': a capitalized word,
    one or more single-letter initials each with a period, and a capitalized
    word of 2+ letters. A naming convention used for people, which is why
    CorroborationValidator trusts it and _select_canonical_span widens to it."""
    tokens = text.split()
    return (len(tokens) >= 3 and bool(_NAME_WORD_RE.match(tokens[0])) and bool(_NAME_WORD_RE.match(tokens[-1]))
            and all(_INITIAL_RE.match(t) for t in tokens[1:-1]))


def _is_plausible_name_token(token: str, knowledge_base: KnowledgeBase) -> bool:
    """True if a capitalized token beside a one-word dictionary name can
    be the rest of that name: an unambiguous known first/last name, or a
    word no list knows that isn't ordinary English, a stopword, a calendar
    word, a language/region name, a place or an organization (i.e. an unseen surname/first name)."""
    word = _strip_possessive(token)[0].rstrip(".,;:")
    if len(word) < 2:
        return False
    if (knowledge_base.is_ambiguous_first_name(word) or knowledge_base.is_calendar_word(word)
            or knowledge_base.is_language_or_region_name(word)):
        return False
    if knowledge_base.is_stopword(word) or knowledge_base.is_location(word) or knowledge_base.is_organization(word):
        return False
    if knowledge_base.is_known_first_name(word) or knowledge_base.is_known_last_name(word):
        return True
    return not knowledge_base.is_common_word(word)


def normalize_name(text: str) -> str:
    """Canonical form used for deduplication/dictionary lookups: collapse
    whitespace, strip trailing punctuation, preserve original casing."""
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.strip(".,;:()[]{}\"'-")


def merge_detections(detections: list[Detection]) -> list[list[Detection]]:
    """Group detections whose spans overlap into clusters. Detections on
    different pages are never merged together."""
    if not detections:
        return []

    by_page: dict[int, list[Detection]] = {}
    for d in detections:
        by_page.setdefault(d.page_index, []).append(d)

    groups: list[list[Detection]] = []
    for page_detections in by_page.values():
        ordered = sorted(page_detections, key=lambda d: (d.start, -d.end))
        current_group: list[Detection] = []
        current_end = -1

        for det in ordered:
            if current_group and det.start < current_end:
                current_group.append(det)
                current_end = max(current_end, det.end)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [det]
                current_end = det.end

        if current_group:
            groups.append(current_group)

    return groups


class CandidateFactory:
    """Builds CandidateResult objects from raw detections, attaching a
    human-readable TextLocation to every candidate via the LineIndex."""

    def __init__(self, line_index: LineIndex, knowledge_base: KnowledgeBase | None = None) -> None:
        self._line_index = line_index
        # Optional: enables the dictionary-single-token boundary rescue in
        # _select_canonical_span; without it that rescue is skipped.
        self._knowledge_base = knowledge_base

    def build(self, detections: list[Detection]) -> list[CandidateResult]:
        groups = merge_detections(detections)
        results: list[CandidateResult] = []

        for group in groups:
            page_index = group[0].page_index

            # Pick the most boundary-trustworthy detection's span as
            # canonical, not just whichever detection happens to be
            # widest - a naive widest-wins policy lets a greedy,
            # grammar-blind regex match override a more precise
            # spaCy/titled boundary whenever they overlap but disagree
            # on the exact extent. See _select_canonical_span for the
            # one narrow, verified exception.
            widest = _select_canonical_span(group, self._knowledge_base)
            start = widest.start
            # Trim a trailing possessive regardless of which detector's
            # span won boundary priority - same fix as
            # dictionary_detector.py already applies to its own matches.
            # Found via real testing: RegexDetector alone stripping this
            # wasn't enough, because spaCy independently tags the SAME
            # possessive-inclusive span ("Wrenna Solkiewicz's") as its
            # own PERSON entity, and spaCy's boundary priority (see
            # _PATTERN_BOUNDARY_PRIORITY above) wins regardless of the
            # regex detector's already-trimmed, narrower span - so the
            # untrimmed spaCy span kept reaching normalize_name() either
            # way, still fragmenting one real person's mentions into two
            # separately-aggregated report entries ("Wrenna Solkiewicz"
            # and "Wrenna Solkiewicz's").
            stemmed_text, trim = _strip_possessive(widest.text.strip())
            end = widest.end - trim
            normalized = normalize_name(stemmed_text)
            location = self._line_index.locate(start, end)

            candidate = Candidate.new(
                text=stemmed_text,
                normalized_text=normalized,
                start=start,
                end=end,
                page_index=page_index,
                source_detections=tuple(group),
                location=location,
            )

            state = CandidateState()
            for detection in group:
                state.add_evidence(
                    source=detection.detector.value,
                    label=f"{detection.detector.value} match ({detection.metadata.get('pattern', 'n/a')})",
                    score=detection.confidence,
                )

            results.append(CandidateResult(candidate=candidate, state=state))

        return results
