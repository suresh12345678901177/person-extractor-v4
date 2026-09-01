"""Rejects candidates that are implausibly short or long to be a person name."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

MIN_CHARS = 3
MAX_CHARS = 60
MIN_TOKENS = 1
MAX_TOKENS = 5


class LengthValidator(BaseValidator):
    name = "length"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        text = candidate.candidate.normalized_text
        tokens = text.split()

        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            return ValidationResult(
                validator_name=self.name, passed=False, severity="hard",
                message=f"Length {len(text)} outside allowed range [{MIN_CHARS}, {MAX_CHARS}]",
            )

        if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
            return ValidationResult(
                validator_name=self.name, passed=False, severity="hard",
                message=f"Token count {len(tokens)} outside allowed range [{MIN_TOKENS}, {MAX_TOKENS}]",
            )

        return ValidationResult(validator_name=self.name, passed=True, score=0.05, message="Length OK")
