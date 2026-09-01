"""
src.validation.validators.context_validator
===============================================
Inspects the text immediately surrounding a candidate's span:

1. Hard-rejects candidates adjacent to URL/email markup - a capitalized
   word immediately preceded by "http://", "https://", "www." or
   immediately followed by "@" is part of a URL/email, not a name.
2. Soft-boosts candidates immediately preceded by a known occupation
   word ("engineer Suresh", "Dr." handled separately by TitleValidator)
   as weak corroborating evidence of personhood.
"""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.segmenter import following_char, preceding_word
from src.validation.base_validator import BaseValidator

_URL_PREFIXES = ("http://", "https://", "www.")


class ContextValidator(BaseValidator):
    name = "context"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        start = candidate.candidate.start
        end = candidate.candidate.end

        preceding_chunk = document_text[max(0, start - 10):start].lower()
        if any(prefix in preceding_chunk for prefix in _URL_PREFIXES):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Immediately preceded by a URL marker (http://, https://, www.)",
            )

        next_char = following_char(document_text, end)
        if next_char == "@":
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Immediately followed by '@' - looks like an email address",
            )

        prev_word = preceding_word(document_text, start)
        if prev_word and knowledge_base.is_occupation(prev_word):
            return ValidationResult(
                self.name, passed=True, score=0.10,
                message=f"Preceded by occupation word '{prev_word}'",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No context signal")
