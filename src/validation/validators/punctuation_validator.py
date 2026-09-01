"""Rejects candidates containing digits or disallowed punctuation."""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

_DIGIT_RE = re.compile(r"\d")
_DISALLOWED_PUNCT_RE = re.compile(r"[!@#$%^&*()_+=\[\]{}<>/\\|~`]")


class PunctuationValidator(BaseValidator):
    name = "punctuation"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        text = candidate.candidate.normalized_text

        if _DIGIT_RE.search(text):
            return ValidationResult(self.name, passed=False, severity="hard", message="Contains digits")

        if _DISALLOWED_PUNCT_RE.search(text):
            return ValidationResult(
                self.name, passed=False, severity="hard", message="Contains disallowed punctuation",
            )

        return ValidationResult(self.name, passed=True, score=0.05, message="Punctuation OK")
