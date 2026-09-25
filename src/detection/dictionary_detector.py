"""
src.detection.dictionary_detector
====================================
Dictionary-based detector: scans consecutive-capitalized-token windows
and flags those where the tokens are consistent with a known first/last
name pair, using the offline KnowledgeBase.

Also emits a low-confidence SINGLE-TOKEN detection when a lone
capitalized word matches a known first OR last name - this is what lets
the pipeline catch names that only ever appear as a single token in the
text (e.g. "Suresh claimed..."), which no shape-based detector can do.
Because a single word is inherently ambiguous, this is deliberately
scored low enough to land in the REVIEW bucket unless corroborated by
another signal (title, spaCy, repeated occurrence).

Two adversarial-testing fixes carried forward from V3:
- Adjacency is checked char-by-char (never across a newline), so a
  detection can't merge two different lines into one fake name.
- Possessive suffixes ('s / 's) are stripped before dictionary lookup
  AND from the reported span, so "Geetha's" is correctly recognized as
  "Geetha" instead of being invisible to an exact-match lookup.
"""

from __future__ import annotations

import re

from src.core.models import Detection, DetectorName
from src.detection.base_detector import BaseDetector
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.latin import LATIN_LETTERS, LATIN_UPPER

# Shared Latin letter classes (src/preprocessing/latin.py). The old
# ASCII-only [A-Z][a-zA-Z]* matched just the ASCII PREFIX of an accented
# word, so "François" produced a dictionary hit on the fragment "Fran".
#
# A token must be a whole word (2026-09-25): no letter, digit or underscore
# directly before or after it. Without the lookarounds the pattern matched
# known names INSIDE code identifiers - "Handler" out of "clickHandler" /
# "errorHandler" in web pages and dumps was ACCEPTED 146 times in the real
# case scan (every one with zero standalone occurrences in its file).
# RegexDetector's \b already prevented this there.
_TOKEN_RE = re.compile(rf"(?<![{LATIN_LETTERS}0-9_])[{LATIN_UPPER}][{LATIN_LETTERS}'\-]*(?![{LATIN_LETTERS}0-9_])")
_POSSESSIVE_SUFFIX_RE = re.compile(r"['\u2019]s$")


def _strip_possessive(word: str) -> tuple[str, int]:
    """Strip a trailing possessive for lookups/span trimming.
    Returns (stem, trimmed_char_count)."""
    match = _POSSESSIVE_SUFFIX_RE.search(word)
    if match:
        return word[:match.start()], len(word) - match.start()
    return word, 0


class DictionaryDetector(BaseDetector):
    name = DetectorName.DICTIONARY

    MIN_CONFIDENCE = 0.60
    MAX_CONFIDENCE = 0.85
    SINGLE_TOKEN_CONFIDENCE = 0.40

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base

    def detect(self, text: str, page_index: int) -> list[Detection]:
        if not text:
            return []

        tokens = list(_TOKEN_RE.finditer(text))
        detections: list[Detection] = []

        i = 0
        while i < len(tokens):
            window = self._extend_window(text, tokens, i)
            if window:
                start_tok, end_tok = window
                _, trim = _strip_possessive(tokens[end_tok].group())
                eff_end = tokens[end_tok].end() - trim
                span_text = text[tokens[start_tok].start():eff_end]
                confidence = self._score_window(tokens[start_tok:end_tok + 1])
                detections.append(
                    Detection(
                        text=span_text,
                        start=tokens[start_tok].start(),
                        end=eff_end,
                        page_index=page_index,
                        detector=self.name,
                        confidence=confidence,
                        metadata={"pattern": "dictionary"},
                    )
                )
                i = end_tok + 1
            else:
                single = self._single_token_match(tokens[i], page_index)
                if single is not None:
                    detections.append(single)
                i += 1

        return detections

    def _single_token_match(self, token: re.Match, page_index: int) -> Detection | None:
        word, trim = _strip_possessive(token.group())
        is_first = self.kb.is_known_first_name(word)
        is_last = self.kb.is_known_last_name(word)
        if not (is_first or is_last):
            return None

        return Detection(
            text=word,
            start=token.start(),
            end=token.end() - trim,
            page_index=page_index,
            detector=self.name,
            confidence=self.SINGLE_TOKEN_CONFIDENCE,
            metadata={"pattern": "dictionary_single_token"},
        )

    def _extend_window(self, text: str, tokens: list[re.Match], start_idx: int) -> tuple[int, int] | None:
        """Try to build a 2-token window starting at start_idx where the
        two tokens independently fill DISTINCT first-name/last-name roles
        (order-agnostic, e.g. handles 'Kumar Suresh' too - one token as
        last name, the other as first name)."""
        if start_idx + 1 >= len(tokens):
            return None

        first_tok = _strip_possessive(tokens[start_idx].group())[0]
        second_tok = _strip_possessive(tokens[start_idx + 1].group())[0]

        # tokens must be adjacent, separated only by spaces/tabs - never
        # a newline (a name must never span multiple lines)
        if not self._adjacent_same_line(text, tokens[start_idx], tokens[start_idx + 1]):
            return None

        # Each token must independently fill a DIFFERENT role (one as
        # first name, the other as last name) - deliberately NOT the
        # earlier "has_first_name = is_first(A) or is_first(B)" /
        # "has_last_name = is_last(B) or is_last(A)" OR-based check,
        # which let a single dual-purpose token (known as BOTH a first
        # AND a last name, e.g. "Kumar" - confirmed via real casework:
        # "Forensics Kumar", "Message Kumar") satisfy both conditions by
        # itself, treating any adjacent capitalized word - including a
        # pure stopword like "Forensics"/"Message" - as if it completed
        # a genuine name pair, even though it contributed zero name
        # evidence of its own.
        first_is_first_name = self.kb.is_known_first_name(first_tok)
        first_is_last_name = self.kb.is_known_last_name(first_tok)
        second_is_first_name = self.kb.is_known_first_name(second_tok)
        second_is_last_name = self.kb.is_known_last_name(second_tok)

        valid_pair = (first_is_first_name and second_is_last_name) or (
            first_is_last_name and second_is_first_name
        )

        if valid_pair:
            end_idx = start_idx + 1
            if end_idx + 1 < len(tokens) and self._adjacent_same_line(text, tokens[end_idx], tokens[end_idx + 1]):
                third_stem = _strip_possessive(tokens[end_idx + 1].group())[0]
                if self.kb.is_known_last_name(third_stem):
                    end_idx += 1
            return start_idx, end_idx

        return None

    @staticmethod
    def _adjacent_same_line(text: str, first: re.Match, second: re.Match) -> bool:
        # Capped at 1-2 gap characters AND the gap must be PURELY
        # space/tab - same tolerance and rationale as regex_detector.py's
        # _SAFE_SEP: tolerates one accidental extra space in real text
        # ("Donald  Richard") without bridging 3+ spaces of deliberate
        # table-column padding. The content check matters as much as the
        # length cap: a length-only check would also treat ", " (comma +
        # space, between two different names in a list - "Anil, Prasad,
        # Naresh") as "adjacent", wrongly letting the detector bridge
        # across a list-item boundary that was never whitespace at all.
        gap_text = text[first.end():second.start()]
        return 0 <= len(gap_text) <= 2 and gap_text.strip(" \t") == "" and "\n" not in gap_text

    def _score_window(self, window_tokens: list[re.Match]) -> float:
        known_count = sum(
            1 for t in window_tokens
            if self.kb.is_known_first_name(_strip_possessive(t.group())[0])
            or self.kb.is_known_last_name(_strip_possessive(t.group())[0])
        )
        ratio = known_count / len(window_tokens)
        return self.MIN_CONFIDENCE + ratio * (self.MAX_CONFIDENCE - self.MIN_CONFIDENCE)
