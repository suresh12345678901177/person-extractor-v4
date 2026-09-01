"""Handles single-letter initials (e.g. 'J. R. Smith'). Requires at
least one full (non-initial) token - an all-initials span ('J. R.') is
not a usable person name on its own."""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

_INITIAL_RE = re.compile(r"^[A-Z]\.?$")


class InitialValidator(BaseValidator):
    name = "initial"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()
        initials = [t for t in tokens if _INITIAL_RE.match(t)]
        full_tokens = [t for t in tokens if not _INITIAL_RE.match(t)]

        if initials and not full_tokens:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Candidate consists only of initials",
            )

        if initials:
            return ValidationResult(
                self.name, passed=True, score=0.05,
                message=f"{len(initials)} initial(s) alongside {len(full_tokens)} full token(s)",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No initials present")
