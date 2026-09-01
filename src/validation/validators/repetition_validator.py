"""Rejects candidates where the same token repeats (e.g. OCR/typo
artifacts like 'John John John') - not a plausible person name."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator


class RepetitionValidator(BaseValidator):
    name = "repetition"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = [t.lower() for t in candidate.candidate.normalized_text.split()]

        if len(tokens) != len(set(tokens)):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Candidate contains repeated tokens",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No repetition")
