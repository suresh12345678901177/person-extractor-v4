"""Hard-rejects candidates matching a known non-person blacklist entry,
checked bidirectionally as sub-phrases (e.g. 'The United Nations' is
still caught even though the blacklist entry is 'United Nations')."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator


class BlacklistValidator(BaseValidator):
    name = "blacklist"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        text = candidate.candidate.normalized_text

        if knowledge_base.is_blacklisted(text):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"'{text}' matches blacklist",
            )

        tokens = text.split()
        for start in range(len(tokens)):
            for end in range(start + 1, len(tokens) + 1):
                sub_phrase = " ".join(tokens[start:end])
                if sub_phrase == text:
                    continue
                if knowledge_base.is_blacklisted(sub_phrase):
                    return ValidationResult(
                        self.name, passed=False, severity="hard",
                        message=f"Contains known non-person phrase: '{sub_phrase}'",
                    )

        return ValidationResult(self.name, passed=True, score=0.0, message="Not blacklisted")
