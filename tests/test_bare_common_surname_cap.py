"""Tests for person_extractor._cap_bare_common_surnames: an ACCEPTED single
ordinary-English word known only as a surname, backed by nothing but that
dictionary hit, is held for REVIEW - and every other shape is left alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Decision, Detection, DetectorName
from src.extraction.person_extractor import _cap_bare_common_surnames
from src.knowledge.knowledge_base import KnowledgeBase

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _accepted(text: str, *patterns: str) -> CandidateResult:
    detections = tuple(
        Detection(text=text, start=0, end=len(text), page_index=0,
                  detector=DetectorName.SPACY if p == "spacy_ner" else DetectorName.DICTIONARY,
                  confidence=0.4, metadata={"pattern": p})
        for p in patterns
    )
    cr = CandidateResult(candidate=Candidate.new(text, text, 0, len(text), 0, detections))
    cr.state.decision = Decision.ACCEPTED
    return cr


def test_word_lists_are_what_the_rule_assumes():
    # If an asset edit changes these, the tests below stop meaning anything.
    assert KB.is_common_word("price") and KB.is_known_last_name("price") and not KB.is_known_first_name("price")
    assert KB.is_common_word("read") and KB.is_known_last_name("read") and not KB.is_known_first_name("read")
    assert KB.is_known_first_name("amber")


def test_bare_common_surname_with_only_a_dictionary_hit_goes_to_review():
    price = _accepted("Price", "dictionary_single_token")
    _cap_bare_common_surnames([price], KB, [])
    assert price.state.decision == Decision.REVIEW
    assert "surname" in price.state.rejection_reason
    assert any("Capped at review" in e.label for e in price.state.evidence)  # visible in --explain


def test_full_name_in_the_same_document_keeps_it_accepted():
    price, full = _accepted("Price", "dictionary_single_token"), _accepted("Dr. Alan Price", "titled")
    _cap_bare_common_surnames([price, full], KB, [])
    assert price.state.decision == Decision.ACCEPTED


def test_full_name_from_earlier_in_a_session_keeps_it_accepted():
    price = _accepted("Price", "dictionary_single_token")
    _cap_bare_common_surnames([price], KB, ["Alan Price"])
    assert price.state.decision == Decision.ACCEPTED


def test_controls_other_evidence_or_shapes_are_left_alone():
    with_spacy = _accepted("Price", "dictionary_single_token", "spacy_ner")
    titled = _accepted("Price", "titled")
    first_name = _accepted("Amber", "dictionary_single_token")   # common word, but a first name
    multi = _accepted("Read Price", "dictionary_single_token")
    review = _accepted("Price", "dictionary_single_token")
    review.state.decision = Decision.REVIEW
    _cap_bare_common_surnames([with_spacy, titled, first_name, multi, review], KB, [])
    assert [c.state.decision for c in (with_spacy, titled, first_name, multi)] == [Decision.ACCEPTED] * 4
    assert review.state.rejection_reason is None  # untouched, not re-labelled
