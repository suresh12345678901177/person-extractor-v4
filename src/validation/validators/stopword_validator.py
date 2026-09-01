"""Hard-rejects a candidate if ANY token is a common stopword (e.g.
'The CCTV') - a genuine person name never contains an English function
word as one of its tokens."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator


class StopwordValidator(BaseValidator):
    name = "stopword"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()
        stopword_tokens = [t for t in tokens if knowledge_base.is_stopword(t)]

        if stopword_tokens:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"Contains stopword token(s): {', '.join(stopword_tokens)}",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No stopword tokens")
