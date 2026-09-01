"""Unit tests for feature extraction and the evaluation metrics module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.evaluation.metrics import compute_span_metrics
from src.features.feature_extractor import extract_features
from src.features.feature_vector import FEATURE_NAMES
from src.knowledge.knowledge_base import KnowledgeBase

KB = KnowledgeBase.load(BASE_DIR / "assets")


def test_feature_vector_field_count_matches_names():
    text = "Dr. Suresh Kumar attended the meeting."
    name = "Dr. Suresh Kumar"
    det = (Detection(text=name, start=0, end=len(name), page_index=0,
                      detector=DetectorName.REGEX, confidence=0.9,
                      metadata={"pattern": "titled"}),)
    candidate = Candidate.new(name, name, 0, len(name), 0, det)
    cr = CandidateResult(candidate=candidate)
    fv = extract_features(cr, text, KB)
    assert len(fv.as_list()) == len(FEATURE_NAMES)


def test_feature_extractor_detects_title_and_dict_hits():
    text = "Dr. Suresh Kumar attended the meeting."
    name = "Dr. Suresh Kumar"
    det = (Detection(text=name, start=0, end=len(name), page_index=0,
                      detector=DetectorName.REGEX, confidence=0.9,
                      metadata={"pattern": "titled"}),)
    candidate = Candidate.new(name, name, 0, len(name), 0, det)
    cr = CandidateResult(candidate=candidate)
    fv = extract_features(cr, text, KB)
    assert fv.has_title == 1
    # "Kumar" is a known LAST name, and separately also a known FIRST
    # name (both real-world usages exist, and this project's
    # first_names.txt now includes it via the India/Sri Lanka
    # firstname-database bulk import) - so it legitimately hits both
    # dictionaries, not just one.
    assert fv.first_name_dict_hits == 2
    assert fv.last_name_dict_hits == 1
    assert fv.token_count == 3


def test_compute_span_metrics_perfect_match():
    predicted = [(0, 5), (10, 15)]
    gold = [(0, 5), (10, 15)]
    m = compute_span_metrics(predicted, gold)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_compute_span_metrics_false_positive():
    predicted = [(0, 5), (20, 25)]
    gold = [(0, 5)]
    m = compute_span_metrics(predicted, gold)
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.precision == 0.5
    assert m.recall == 1.0


def test_compute_span_metrics_false_negative():
    predicted = [(0, 5)]
    gold = [(0, 5), (20, 25)]
    m = compute_span_metrics(predicted, gold)
    assert m.true_positives == 1
    assert m.false_negatives == 1
    assert m.recall == 0.5


def test_compute_span_metrics_overlap_counts_as_match():
    # A predicted span that partially overlaps a gold span (e.g. with/
    # without a title prefix) should still count as a hit.
    predicted = [(4, 20)]   # "Suresh Kumar" within "Dr. Suresh Kumar"
    gold = [(0, 20)]        # "Dr. Suresh Kumar"
    m = compute_span_metrics(predicted, gold)
    assert m.true_positives == 1
