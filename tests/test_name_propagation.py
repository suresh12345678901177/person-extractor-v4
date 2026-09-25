"""Tests for document-level first-name propagation
(src/candidate/name_propagation.py) and its pipeline integration."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import BASE_DIR, DEFAULT_CONFIG
from src.candidate.name_propagation import (
    DOCUMENT_NAME_PATTERN,
    propagatable_first_names,
    undetected_occurrences,
)
from src.core.models import (
    Candidate, CandidateResult, ClassifierResult, Decision, Detection, DetectorName,
)
from src.decision.decision_engine import DecisionEngine
from src.knowledge.knowledge_base import KnowledgeBase

KB = KnowledgeBase.load(BASE_DIR / "assets")


@pytest.fixture(scope="module")
def pipeline():
    from src.pipeline.orchestrator import Pipeline

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    return Pipeline(config=config, base_dir=BASE_DIR)


def _patterns(mention) -> set[str]:
    return {d.metadata.get("pattern") for d in mention.candidate.source_detections}


def test_first_name_of_accepted_full_name_is_accepted_alone(pipeline):
    """'Kenji' is in no name list and nothing tags it on its own - but the
    same document accepted 'Dr. Kenji Ito'."""
    for text in (
        "Dr. Kenji Ito signed the report on Monday.\nKenji\nPlease review the numbers before Friday.\n",
        "Dr. Kenji Ito signed the report on Monday. Later that day, we asked Kenji about the invoice.\n",
    ):
        result = pipeline.run_text(text)
        kenji = [m for p in result.persons for m in p.mentions if m.candidate.text == "Kenji"]
        assert kenji, [p.display_text for p in result.persons]
        assert DOCUMENT_NAME_PATTERN in _patterns(kenji[0])


def test_no_propagation_without_an_accepted_full_name(pipeline):
    result = pipeline.run_text("Kenji\nPlease review the numbers before Friday.\n")
    assert not any(m.candidate.text == "Kenji" for p in result.persons for m in p.mentions)


def test_common_and_ambiguous_first_names_are_never_propagated():
    names = propagatable_first_names(["Chase Miller", "Grace Okafor", "Dr. Kenji Ito", "J. Smith"], KB)
    assert names == {"Kenji": "Kenji Ito"}


def test_first_half_of_an_already_covered_name_is_not_split_off():
    """'Ranjodh Aulakh' where only 'Aulakh' was detected: adding 'Ranjodh'
    separately would fragment one name into two mentions."""
    text = "Later Ranjodh Aulakh left early."
    start = text.index("Aulakh")
    aulakh = CandidateResult(candidate=Candidate.new(
        "Aulakh", "Aulakh", start, start + 6, 0,
        (Detection("Aulakh", start, start + 6, 0, DetectorName.DICTIONARY, 0.4, {"pattern": "dictionary_single_token"}),),
    ))
    assert undetected_occurrences(text, {"Ranjodh": "Ranjodh Aulakh"}, [aulakh]) == []
    # Standalone, it IS found.
    assert len(undetected_occurrences("Ask Ranjodh tomorrow.", {"Ranjodh": "Ranjodh Aulakh"}, [])) == 1


def test_decide_is_safe_to_call_twice():
    """Propagation re-decides candidates; ML evidence must not be counted twice."""
    text = "Priya Raman arrived."
    det = Detection("Priya Raman", 0, 11, 0, DetectorName.DICTIONARY, 0.85, {"pattern": "dictionary"})
    cr = CandidateResult(candidate=Candidate.new("Priya Raman", "Priya Raman", 0, 11, 0, (det,)))
    cr.state.add_evidence("dictionary", "dictionary match", 0.85)
    cr.state.classifier_result = ClassifierResult(label=1, probability=0.9, model_name="m", model_version="1")
    engine = DecisionEngine()
    engine.decide(cr, text, KB)
    first = cr.state.final_confidence
    engine.decide(cr, text, KB)
    assert cr.state.final_confidence == first
    assert sum(1 for e in cr.state.evidence if e.source == "ml_classifier") == 1
    assert cr.state.decision == Decision.ACCEPTED
