"""
src.validation.base_validator
===============================
Base interface for the Validation Firewall. Each validator has exactly
one responsibility and returns a ValidationResult (never a bare
boolean) - per the blueprint's Design Principles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase


class BaseValidator(ABC):
    name: str

    @abstractmethod
    def validate(
        self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str
    ) -> ValidationResult:
        """document_text is the full cleaned document text, passed so
        context-dependent validators (Grammar, Context) can inspect the
        text surrounding the candidate's span without each validator
        needing its own way of accessing it."""
        raise NotImplementedError
