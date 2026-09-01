"""Soft-scores candidates higher when their tokens appear in the offline
first-name/last-name knowledge base."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator


class DictionaryValidator(BaseValidator):
    name = "dictionary"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()
        known = sum(
            1 for t in tokens
            if knowledge_base.is_known_first_name(t) or knowledge_base.is_known_last_name(t)
        )

        if known == 0:
            return ValidationResult(
                self.name, passed=True, score=0.0,
                message="No tokens found in name dictionaries",
            )

        ratio = known / len(tokens)
        return ValidationResult(
            self.name, passed=True, score=round(0.70 * ratio, 2),
            message=f"{known}/{len(tokens)} tokens matched name dictionaries",
        )
