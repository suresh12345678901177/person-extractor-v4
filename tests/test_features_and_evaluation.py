"""Unit tests for feature extraction and the evaluation metrics module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.evaluation.evaluator import _gold_mention_shapes
from src.evaluation.metrics import RecallBreakdown, compute_span_metrics
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
    # dictionaries, not just one. Same story for "Suresh" as a LAST
    # name after the 2026-09-16 global (Faker, 81 Latin-script locales)
    # bulk import - en_IN-derived data includes it there too, consistent
    # with South Indian naming conventions where a given name can also
    # function as a surname.
    assert fv.first_name_dict_hits == 2
    assert fv.last_name_dict_hits == 2
    assert fv.token_count == 3


def test_feature_extractor_detects_preceding_person_cue():
    text = "According to Marcus Webb, the timeline needed to change."
    name = "Marcus Webb"
    start = text.index(name)
    det = (Detection(text=name, start=start, end=start + len(name), page_index=0,
                      detector=DetectorName.REGEX, confidence=0.55,
                      metadata={"pattern": "bare"}),)
    candidate = Candidate.new(name, name, start, start + len(name), 0, det)
    cr = CandidateResult(candidate=candidate)
    fv = extract_features(cr, text, KB)
    assert fv.preceding_context_is_person_cue == 1
    assert fv.following_context_is_nonperson_cue == 0


def test_feature_extractor_detects_following_nonperson_cue():
    text = "The invoice was issued by Marcus Webb Ltd last month."
    name = "Marcus Webb"
    start = text.index(name)
    det = (Detection(text=name, start=start, end=start + len(name), page_index=0,
                      detector=DetectorName.REGEX, confidence=0.55,
                      metadata={"pattern": "bare"}),)
    candidate = Candidate.new(name, name, start, start + len(name), 0, det)
    cr = CandidateResult(candidate=candidate)
    fv = extract_features(cr, text, KB)
    assert fv.following_context_is_nonperson_cue == 1


def test_feature_extractor_detects_isolated_line():
    text = "Marcus Webb\nHey, are you free to talk later today?"
    name = "Marcus Webb"
    det = (Detection(text=name, start=0, end=len(name), page_index=0,
                      detector=DetectorName.REGEX, confidence=0.55,
                      metadata={"pattern": "bare"}),)
    candidate = Candidate.new(name, name, 0, len(name), 0, det)
    cr = CandidateResult(candidate=candidate)
    fv = extract_features(cr, text, KB)
    assert fv.is_isolated_line == 1


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


def test_recall_breakdown_computes_ratio():
    rb = RecallBreakdown(hits=3, total=4)
    assert rb.recall == 0.75


def test_recall_breakdown_zero_total_is_zero_not_error():
    rb = RecallBreakdown(hits=0, total=0)
    assert rb.recall == 0.0


def test_gold_mention_shapes_single_token_dictionary_known():
    token_shape, dict_shape = _gold_mention_shapes("Suresh", KB)
    assert token_shape == "single_token"
    assert dict_shape == "dictionary_known"


def test_gold_mention_shapes_multi_token_unseen():
    token_shape, dict_shape = _gold_mention_shapes("Xyzlq Vbnmpqr", KB)
    assert token_shape == "multi_token"
    assert dict_shape == "unseen_name"


def test_text_only_features_separate_plain_words_from_names():
    from src.features.feature_extractor import text_only_features

    phrase_ratio, _, _ = text_only_features("Performance Review", KB)
    name_ratio, _, _ = text_only_features("Nadia Kovalenko", KB)
    assert phrase_ratio == 1.0
    assert name_ratio < phrase_ratio


def test_text_only_features_ignore_titles():
    """'Dr.' must not count as an ordinary word diluting the ratio."""
    from src.features.feature_extractor import text_only_features

    assert text_only_features("Dr. Nadia Kovalenko", KB) == text_only_features("Nadia Kovalenko", KB)


def test_extract_features_carries_text_only_features():
    from src.features.feature_extractor import text_only_features

    text = "The Deep Tunnel was flooded."
    candidate = CandidateResult(candidate=Candidate.new(
        "Deep Tunnel", "Deep Tunnel", 4, 15, 0,
        (Detection("Deep Tunnel", 4, 15, 0, DetectorName.REGEX, 0.55, {"pattern": "bare"}),),
    ))
    fv = extract_features(candidate, text, KB)
    assert (fv.common_word_ratio, fv.first_token_is_ambiguous, fv.place_or_org_token_count) == \
        text_only_features("Deep Tunnel", KB)


def test_pooled_benchmark_metrics_never_match_spans_across_files(tmp_path):
    """File A has a gold mention at chars 0-5 that the pipeline misses;
    file B has a false positive at the SAME offsets. Pooling bare offsets
    used to count that as a true positive. Overall must equal the sum of
    the per-file results: 0 TP, 1 FP, 1 FN."""
    import json
    from types import SimpleNamespace

    from src.evaluation.evaluator import run_benchmark

    for name, gold in (("a_case.txt", [{"start": 0, "end": 5, "text": "Priya"}]), ("b_case.txt", [])):
        (tmp_path / name).write_text("Priya went home.", encoding="utf-8")
        (tmp_path / name.replace(".txt", "_gold.json")).write_text(json.dumps({"mentions": gold}), encoding="utf-8")

    def fake_run(path):
        mentions = [SimpleNamespace(candidate=SimpleNamespace(start=0, end=5))] if "b_case" in str(path) else []
        persons = (SimpleNamespace(mentions=mentions),) if mentions else ()
        return SimpleNamespace(persons=persons, review_persons=())

    pipeline = SimpleNamespace(run=fake_run, extractor=SimpleNamespace(knowledge_base=KB))
    overall = run_benchmark(pipeline, tmp_path).overall_accepted
    assert (overall.true_positives, overall.false_positives, overall.false_negatives) == (0, 1, 1)
