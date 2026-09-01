"""
src.validation.validators.language_validator
================================================
Two checks, both defensive against non-English text being misread as a
capitalized English name:

1. Non-Latin script guard: if a candidate somehow contains a character
   outside the Latin-plus-punctuation range (in practice rare, since the
   regex/dictionary detectors' token pattern is ASCII-letters-only by
   construction - but this validator is deliberately defensive in case a
   future detector or Unicode-confusable input changes that assumption).
2. Transliterated marker-word guard: rejects a candidate whose token
   matches a known transliterated non-English function word (e.g. "Ra",
   "Naa" - Telugu words that romanize to look like a capitalized English
   name-initial token at a sentence boundary). List lives in
   `assets/languages/non_english_markers.txt`.
"""

from __future__ import annotations

import unicodedata

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator


def _is_latin_or_punct(ch: str) -> bool:
    if ch.isspace():
        return True
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return "LATIN" in name or not ch.isalpha()


class LanguageValidator(BaseValidator):
    name = "language"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        text = candidate.candidate.normalized_text

        non_latin = [ch for ch in text if ch.isalpha() and not _is_latin_or_punct(ch)]
        if non_latin:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"Contains non-Latin script character(s): {set(non_latin)}",
            )

        tokens = text.split()
        marker_tokens = [t for t in tokens if t.lower().rstrip(".") in knowledge_base.non_english_markers]
        if marker_tokens:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"Contains transliterated non-English marker word(s): {', '.join(marker_tokens)}",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No language markers")
