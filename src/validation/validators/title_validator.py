"""Soft-boosts candidates whose supporting detections were found via a
titled pattern (e.g. 'Dr. Suresh Kumar') - a strong signal of personhood."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator


class TitleValidator(BaseValidator):
    name = "title"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        has_title_evidence = any(
            d.metadata.get("pattern") == "titled"
            for d in candidate.candidate.source_detections
        )

        if has_title_evidence:
            return ValidationResult(
                self.name, passed=True, score=0.30,
                message="Preceded by a recognized title/honorific",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No title evidence")
