"""Unit tests for the 15-validator Validation Firewall."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.validators.blacklist_validator import BlacklistValidator
from src.validation.validators.campaign_validator import CampaignValidator
from src.validation.validators.context_validator import ContextValidator
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

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _candidate(text: str, pattern: str = "bare") -> CandidateResult:
    detection = Detection(
        text=text, start=0, end=len(text), page_index=0,
        detector=DetectorName.REGEX, confidence=0.9, metadata={"pattern": pattern},
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, (detection,))
    return CandidateResult(candidate=candidate)


def _validate(validator, text: str, document_text: str | None = None, pattern: str = "bare"):
    cand = _candidate(text, pattern=pattern)
    return validator.validate(cand, KB, document_text if document_text is not None else text)


def test_length_validator_rejects_too_short():
    assert _validate(LengthValidator(), "Al").passed is False


def test_length_validator_accepts_normal_name():
    assert _validate(LengthValidator(), "John Smith").passed is True


def test_structure_validator_rejects_lowercase_token():
    assert _validate(StructureValidator(), "john smith").passed is False


def test_structure_validator_rejects_acronym():
    assert _validate(StructureValidator(), "CCTV Footage").passed is False


def test_structure_validator_rejects_hyphenated_acronym_fragment():
    assert _validate(StructureValidator(), "IDs TXN").passed is False


def test_punctuation_validator_rejects_digits():
    assert _validate(PunctuationValidator(), "John5 Smith").passed is False


def test_repetition_validator_rejects_repeated_tokens():
    assert _validate(RepetitionValidator(), "John John").passed is False


def test_initial_validator_rejects_all_initials():
    assert _validate(InitialValidator(), "J. R.").passed is False


def test_initial_validator_accepts_initial_plus_full_token():
    assert _validate(InitialValidator(), "J. Smith").passed is True


def test_blacklist_validator_rejects_known_entry():
    assert _validate(BlacklistValidator(), "New York").passed is False


def test_blacklist_validator_rejects_phrase_with_leading_article():
    result = _validate(BlacklistValidator(), "The United Nations")
    assert result.passed is False


def test_blacklist_validator_accepts_normal_name():
    assert _validate(BlacklistValidator(), "John Smith").passed is True


def test_campaign_validator_rejects_known_slogan():
    assert _validate(CampaignValidator(), "Jai Telangana").passed is False


def test_campaign_validator_rejects_novel_slogan_via_marker_word():
    assert _validate(CampaignValidator(), "Bharat Mata Ki Jai").passed is False


def test_campaign_validator_accepts_normal_name():
    assert _validate(CampaignValidator(), "Suresh Kumar").passed is True


def test_organization_validator_rejects_business_name():
    assert _validate(OrganizationValidator(), "Sri Lakshmi Ganesh Traders").passed is False


def test_organization_validator_accepts_normal_name():
    assert _validate(OrganizationValidator(), "Suresh Kumar").passed is True


def test_location_validator_rejects_known_city():
    assert _validate(LocationValidator(), "Hyderabad").passed is False


def test_location_validator_accepts_normal_name():
    assert _validate(LocationValidator(), "Suresh Kumar").passed is True


def test_stopword_validator_rejects_any_stopword_token():
    assert _validate(StopwordValidator(), "The CCTV").passed is False


def test_language_validator_rejects_non_latin_script():
    assert _validate(LanguageValidator(), "\u0930\u093e \u0905\u0928\u094d\u0928\u093e").passed is False


def test_language_validator_accepts_normal_name():
    assert _validate(LanguageValidator(), "Suresh Kumar").passed is True


def test_title_validator_boosts_titled_pattern():
    result = _validate(TitleValidator(), "Suresh Kumar", pattern="titled")
    assert result.score > 0


def test_grammar_validator_rejects_sentence_initial_unsupported_single_token():
    doc = "The meeting ended. Nonsense was reported by nobody."
    cand = _candidate("Nonsense")
    # override candidate span to point at the real sentence-initial position
    from dataclasses import replace
    start = doc.index("Nonsense")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Nonsense"), source_detections=())
    result = GrammarValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_grammar_validator_exempts_dictionary_confirmed_single_token():
    doc = "Suresh claimed he heard shouting outside."
    detection = Detection(text="Suresh", start=0, end=6, page_index=0,
                           detector=DetectorName.DICTIONARY, confidence=0.4,
                           metadata={"pattern": "dictionary_single_token"})
    candidate = Candidate.new("Suresh", "Suresh", 0, 6, 0, (detection,))
    cand = CandidateResult(candidate=candidate)
    result = GrammarValidator().validate(cand, KB, doc)
    assert result.passed is True


def test_context_validator_rejects_url_adjacency():
    doc = "Visit https://Example.com for details."
    start = doc.index("Example")
    cand = _candidate("Example")
    from dataclasses import replace
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Example"))
    result = ContextValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_context_validator_boosts_occupation_precedence():
    doc = "The engineer Suresh reviewed the design."
    start = doc.index("Suresh")
    cand = _candidate("Suresh")
    from dataclasses import replace
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Suresh"))
    result = ContextValidator().validate(cand, KB, doc)
    assert result.passed is True and result.score > 0
