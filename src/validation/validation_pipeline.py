"""
src.validation.validation_pipeline
=====================================
Runs every registered validator against a CandidateResult, collecting
all ValidationResults into the candidate's state, and recording evidence
for each validator that contributed a non-zero score.

Hard validators run first (cheap, high-signal rejects) and short-circuit
on the first hard failure - a rejected candidate skips remaining
soft-scoring work, and the failing validator's message becomes the
rejection reason shown in the final report.

This is V4's full 16-validator Validation Firewall: the blueprint's
original 15 (Structure, Length, Grammar, Dictionary, Blacklist,
Stopword, Language, Campaign, Organization, Location, Title, Initial,
Context, Punctuation, Repetition) plus Corroboration, added after
real-world testing showed bare shape-matches with zero other evidence
need to be rejected outright, not merely left for a human to sift
through (see corroboration_validator.py for the full story).
"""

from __future__ import annotations

from src.core.models import CandidateResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.utils.logger import get_logger
from src.validation.base_validator import BaseValidator
from src.validation.validators.blacklist_validator import BlacklistValidator
from src.validation.validators.campaign_validator import CampaignValidator
from src.validation.validators.context_validator import ContextValidator
from src.validation.validators.corroboration_validator import CorroborationValidator
from src.validation.validators.dictionary_validator import DictionaryValidator
from src.validation.validators.grammar_validator import GrammarValidator
from src.validation.validators.initial_validator import InitialValidator
from src.validation.validators.language_validator import LanguageValidator
from src.validation.validators.length_validator import LengthValidator
from src.validation.validators.location_validator import LocationValidator
from src.validation.validators.organization_validator import OrganizationValidator
from src.validation.validators.punctuation_validator import PunctuationValidator
from src.validation.validators.repetition_validator import RepetitionValidator
from src.validation.validators.stopword_validator import StopwordValidator
from src.validation.validators.structure_validator import StructureValidator
from src.validation.validators.title_validator import TitleValidator

logger = get_logger("validation.validation_pipeline")


class ValidationPipeline:
    def __init__(self, strict_corroboration: bool = True) -> None:
        # Hard/structural validators: cheap, unambiguous, high-signal
        # rejects. Ordered roughly cheapest-and-most-common-rejection
        # first, so the average candidate is filtered as fast as possible.
        self.hard_validators: list[BaseValidator] = [
            LengthValidator(),
            StructureValidator(),
            PunctuationValidator(),
            RepetitionValidator(),
            InitialValidator(),
            LanguageValidator(),
        ]
        if strict_corroboration:
            # Comparison/testing knob (see cli.py's --loose-gate): when
            # False, bare shape-matches with zero dictionary/title/spaCy
            # support are no longer hard-rejected outright - they fall
            # through to the Decision Engine's normal scoring instead,
            # which then also needs allow_spacy_only_corroboration=True
            # (see DecisionEngine) for spaCy-only matches to have any
            # real chance of reaching ACCEPTED. OFF by default - this
            # validator's existence is itself the documented, deliberate
            # precision safeguard (see corroboration_validator.py).
            self.hard_validators.append(CorroborationValidator())
        self.hard_validators.extend([
            BlacklistValidator(),
            CampaignValidator(),
            OrganizationValidator(),
            LocationValidator(),
            StopwordValidator(),
            GrammarValidator(),
            ContextValidator(),  # also contributes a soft score when it passes
        ])
        # Pure soft/scoring validators - only run if the candidate
        # survives every hard validator above.
        self.soft_validators: list[BaseValidator] = [
            DictionaryValidator(),
            TitleValidator(),
        ]

    def run(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> CandidateResult:
        for validator in self.hard_validators:
            try:
                result = validator.validate(candidate, knowledge_base, document_text)
            except Exception:
                logger.exception("Validator %s raised on candidate '%s'",
                                  validator.name, candidate.text)
                continue

            candidate.state.validation_results.append(result)
            if result.score:
                candidate.state.add_evidence(validator.name, result.message, result.score)

            if not result.passed and result.severity == "hard":
                candidate.state.rejection_reason = f"{validator.name}: {result.message}"
                return candidate  # short-circuit; still fully explainable

        for validator in self.soft_validators:
            try:
                result = validator.validate(candidate, knowledge_base, document_text)
            except Exception:
                logger.exception("Validator %s raised on candidate '%s'",
                                  validator.name, candidate.text)
                continue

            candidate.state.validation_results.append(result)
            if result.score:
                candidate.state.add_evidence(validator.name, result.message, result.score)

        return candidate
