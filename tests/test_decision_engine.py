"""Unit tests for DecisionEngine's knowledge-corroboration gate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, ClassifierResult, Decision, Detection, DetectorName, FeatureVector
from src.decision.decision_engine import DecisionEngine
from src.knowledge.knowledge_base import KnowledgeBase

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _bare_regex_candidate(text: str, ml_prob: float) -> CandidateResult:
    """A candidate with ONLY bare regex evidence (confidence 0.55) and a
    given ML probability - zero dictionary/title/spaCy support at all,
    the exact shape of the real "Vice President"/"Convertible Debt" bug."""
    detection = Detection(
        text=text, start=0, end=len(text), page_index=0,
        detector=DetectorName.REGEX, confidence=0.55, metadata={"pattern": "bare"},
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, (detection,))
    result = CandidateResult(candidate=candidate)
    result.state.add_evidence("regex", "regex match (bare)", 0.55)
    result.state.classifier_result = ClassifierResult(label=1, probability=ml_prob, model_name="test")
    return result


def test_ml_confidence_alone_does_not_accept_common_word_phrase():
    # Regression test for a real production bug (2026-09-16, "corporate
    # fraud" case scan): a bare, zero-dictionary/title/spaCy candidate
    # made entirely of ordinary English words ("Vice President") auto-
    # ACCEPTED purely on a high ML score + heavy in-document repetition,
    # because nothing checked whether the words themselves meant
    # "person" at all. High ML confidence + repetition must now
    # additionally require the candidate NOT be a plain common-word
    # phrase (see assets/common_words/common_english_words.txt).
    text = "Vice President"
    document_text = " ".join([text] * 6)  # well above MIN_REPETITION_FOR_ML_ONLY_CORROBORATION
    candidate = _bare_regex_candidate(text, ml_prob=0.97)

    DecisionEngine().decide(candidate, document_text, KB)

    assert candidate.state.decision != Decision.ACCEPTED, (
        "A bare common-word phrase must not auto-ACCEPT on ML confidence + repetition alone"
    )


def test_ml_confidence_alone_still_accepts_non_common_word_phrase():
    # Control case: the pre-existing repetition-based ML-confidence
    # corroboration path (see decision_engine.py's third design
    # principle) must still work for a candidate that ISN'T just plain
    # English vocabulary - proves the new guard is targeted at common-
    # word phrases specifically, not a blanket disable of that path.
    text = "Xyzlq Vbnmp"
    document_text = " ".join([text] * 6)
    candidate = _bare_regex_candidate(text, ml_prob=0.97)

    DecisionEngine().decide(candidate, document_text, KB)

    assert candidate.state.decision == Decision.ACCEPTED, (
        "A non-common-word bare phrase with high ML confidence and heavy repetition "
        "should still corroborate via the existing repetition path"
    )


def test_common_word_phrase_with_zero_corroboration_is_rejected_not_reviewed():
    # Regression test for a real production finding (2026-09-18,
    # real case scan): REVIEW kept filling up with plain
    # business-vocabulary noun phrases ("Performance Review", "Retention
    # Bonus", "Corporate Controller", "Coordinated Universal Time",
    # "Deferred Compensation") that scored into the REVIEW range on bare
    # shape evidence alone, with zero dictionary/title/spaCy/ML
    # corroboration ever suggesting they meant "person". These must now
    # be REJECTED outright rather than parked in REVIEW.
    text = "Performance Review"
    document_text = text  # single occurrence - no repetition-based path applies
    candidate = _bare_regex_candidate(text, ml_prob=0.0)

    DecisionEngine().decide(candidate, document_text, KB)

    assert candidate.state.decision == Decision.REJECTED, (
        "A common-word phrase with zero knowledge corroboration must be REJECTED, "
        "not left in REVIEW, no matter how it scored on shape evidence alone"
    )


def test_single_token_dictionary_hit_with_low_ml_corroborates_via_repetition():
    # Regression test for a real diagnosed gap (2026-09-22, targeted
    # multi_token-recall investigation): "Femi Novak" (Femi = genuine
    # first-name dictionary hit, Novak = not in any dictionary) scored ML
    # probability 0.039 on the real benchmark - well under
    # ML_WEAK_DICTIONARY_VETO_FLOOR (0.10) - and stayed capped at REVIEW
    # despite repeating 4+ times in the real document, a real signal the
    # original veto-floor-only check ignored entirely.
    text = "Femi Novak"
    document_text = " ".join([text] * 4)  # meets MIN_REPETITION_FOR_ML_ONLY_CORROBORATION
    detections = (
        Detection(
            text=text, start=0, end=len(text), page_index=0,
            detector=DetectorName.DICTIONARY, confidence=0.5,
            metadata={"pattern": "dictionary_single_token"},
        ),
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, detections)
    result = CandidateResult(candidate=candidate)
    result.state.add_evidence("dictionary", "single-token dictionary hit", 0.5)
    result.state.add_evidence("regex", "regex match (bare)", 0.55)
    result.state.classifier_result = ClassifierResult(label=1, probability=0.04, model_name="test")

    DecisionEngine().decide(result, document_text, KB)

    assert result.state.decision == Decision.ACCEPTED, (
        "A single-token dictionary hit with low ML confidence but real "
        "document-wide repetition (>=4) must now corroborate and reach ACCEPTED"
    )


def test_single_token_dictionary_hit_with_low_ml_and_low_repetition_stays_capped():
    # Control case: the same shape but WITHOUT enough repetition must NOT
    # be newly corroborated - proves the fix is scoped to real repetition,
    # not a blanket floor removal.
    text = "Ilya Zaidi"
    document_text = text  # single occurrence only
    detections = (
        Detection(
            text=text, start=0, end=len(text), page_index=0,
            detector=DetectorName.DICTIONARY, confidence=0.5,
            metadata={"pattern": "dictionary_single_token"},
        ),
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, detections)
    result = CandidateResult(candidate=candidate)
    result.state.add_evidence("dictionary", "single-token dictionary hit", 0.5)
    result.state.add_evidence("regex", "regex match (bare)", 0.55)
    result.state.classifier_result = ClassifierResult(label=1, probability=0.04, model_name="test")

    DecisionEngine().decide(result, document_text, KB)

    assert result.state.decision != Decision.ACCEPTED, (
        "A single-token dictionary hit with low ML confidence and no real "
        "repetition must NOT be newly accepted by this fix"
    )


def _isolated_line_candidate(text: str, ml_prob: float) -> CandidateResult:
    """A zero-dictionary-evidence candidate sitting alone on its own
    line (the chat/email sender-header shape), with a given ML
    probability and feature_vector.is_isolated_line=True set directly -
    the exact shape the 2026-09-22 unseen_name structural-corroboration
    path targets."""
    detection = Detection(
        text=text, start=0, end=len(text), page_index=0,
        detector=DetectorName.REGEX, confidence=0.55, metadata={"pattern": "bare"},
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, (detection,))
    result = CandidateResult(candidate=candidate)
    result.state.add_evidence("regex", "regex match (bare)", 0.55)
    result.state.classifier_result = ClassifierResult(label=1, probability=ml_prob, model_name="test")
    result.state.feature_vector = FeatureVector(
        token_count=len(text.split()), char_length=len(text), has_title=0, has_honorific=0,
        first_name_dict_hits=0, last_name_dict_hits=0, dict_hit_ratio=0.0, capitalized_ratio=1.0,
        initial_token_count=0, regex_confidence=0.55, dictionary_confidence=0.0, spacy_confidence=0.0,
        detector_count=1, preceding_word_is_stopword=0, following_char_is_punct=0, occurs_in_quotes=0,
        preceding_context_is_person_cue=0, following_context_is_nonperson_cue=0,
        is_isolated_line=1,
    )
    return result


def test_isolated_line_unseen_name_with_modest_ml_corroborates_via_repetition():
    # Regression test for a real diagnosed gap (2026-09-22, Priority 2 of
    # the targeted unseen_name investigation): a genuinely rare/unseen
    # name with zero dictionary hit, sitting alone on its own line (the
    # real chat/email sender-header shape), repeating 4+ times, with a
    # modest (not "confident") ML score, was previously stuck at REVIEW
    # forever since neither existing corroboration path covers this
    # shape. "Kenji Ito" scored ml=0.38 on the real benchmark - well
    # above the ML_WEAK_DICTIONARY_VETO_FLOOR guard this path requires.
    text = "Kenji Ito"
    document_text = "\n".join([text] * 12)  # isolated on its own line, 12 occurrences
    candidate = _isolated_line_candidate(text, ml_prob=0.38)

    DecisionEngine().decide(candidate, document_text, KB)

    assert candidate.state.decision == Decision.ACCEPTED, (
        "An isolated-line, zero-dictionary candidate with modest (not confident) ML "
        "score and real repetition must now corroborate via the structural path"
    )


def test_isolated_line_unseen_name_below_ml_floor_stays_capped():
    # Control case: same shape, but ML score below ML_WEAK_DICTIONARY_
    # VETO_FLOOR (the "ML actively disagrees" guard, same protection the
    # 'Dark Corridor' incident needed) must NOT be newly corroborated.
    text = "Dark Corridor"
    document_text = "\n".join([text] * 12)
    candidate = _isolated_line_candidate(text, ml_prob=0.02)

    DecisionEngine().decide(candidate, document_text, KB)

    assert candidate.state.decision != Decision.ACCEPTED, (
        "An isolated-line candidate scoring below the ML floor must not be "
        "accepted by the structural corroboration path"
    )


def test_isolated_line_unseen_name_without_repetition_stays_capped():
    # Control case: same shape and ML score, but only ONE occurrence -
    # must NOT be newly corroborated (repetition is a required signal,
    # not optional).
    text = "Kenji Ito"
    document_text = text  # single occurrence only
    candidate = _isolated_line_candidate(text, ml_prob=0.38)

    DecisionEngine().decide(candidate, document_text, KB)

    assert candidate.state.decision != Decision.ACCEPTED, (
        "An isolated-line candidate with only one occurrence must not be "
        "accepted by the structural corroboration path"
    )


def test_common_word_phrase_with_real_dictionary_corroboration_is_not_force_rejected():
    # Control case: the new REVIEW->REJECTED guard above must only fire
    # when there is genuinely ZERO knowledge corroboration. "Grace" is
    # both an ordinary common word AND a real first name - a candidate
    # with a real single-token dictionary hit on it must be unaffected
    # by the common-word-phrase guard, proving a genuine (if unusual)
    # name that happens to also read as common vocabulary still reaches
    # a human via REVIEW/ACCEPTED instead of being silently discarded.
    text = "Grace Review"
    document_text = text
    detections = (
        Detection(
            text=text, start=0, end=len(text), page_index=0,
            detector=DetectorName.DICTIONARY, confidence=0.6,
            metadata={"pattern": "dictionary_single_token"},
        ),
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, detections)
    result = CandidateResult(candidate=candidate)
    result.state.add_evidence("dictionary", "single-token dictionary hit", 0.6)
    result.state.classifier_result = ClassifierResult(label=1, probability=0.15, model_name="test")

    DecisionEngine().decide(result, document_text, KB)

    assert result.state.decision != Decision.REJECTED, (
        "A common-word phrase with a real single-token dictionary hit must not be "
        "force-rejected by the common-word-phrase guard"
    )
