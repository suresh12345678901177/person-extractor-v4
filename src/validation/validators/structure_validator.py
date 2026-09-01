"""Validates token shape: starts with a capital letter, letters/hyphens/
apostrophes only, and is not an ALL-CAPS acronym or acronym-plural/
hyphenated-acronym-fragment (e.g. 'CCTV', 'IDs', 'TXN-AE')."""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

_VALID_TOKEN_RE = re.compile(r"^[A-Z][a-zA-Z'\-]*\.?$")
_ACRONYM_RE = re.compile(r"^[A-Z]{2,}$")
_ACRONYM_PLURAL_RE = re.compile(r"^[A-Z]{2,}s$")


class StructureValidator(BaseValidator):
    name = "structure"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()

        if not tokens:
            return ValidationResult(self.name, passed=False, severity="hard", message="Empty candidate")

        for token in tokens:
            if not _VALID_TOKEN_RE.match(token):
                return ValidationResult(
                    self.name, passed=False, severity="hard",
                    message=f"Token '{token}' does not match name-token shape",
                )

            letters_only = token.replace("-", "").rstrip(".")
            is_acronym = _ACRONYM_RE.match(letters_only) or _ACRONYM_PLURAL_RE.match(letters_only)
            if is_acronym and not knowledge_base.is_known_last_name(token):
                return ValidationResult(
                    self.name, passed=False, severity="hard",
                    message=f"Token '{token}' looks like an acronym, not a name",
                )

        return ValidationResult(self.name, passed=True, score=0.10, message="Structure OK")
