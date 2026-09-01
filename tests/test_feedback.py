"""Tests for the active learning feedback loop: logging, labeling, and
that retraining data assembly works correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import AggregatedPerson, Candidate, CandidateResult, CandidateState, Decision, Detection, DetectorName
from src.feedback.feedback_store import FeedbackRecord, FeedbackStore


def _fake_review_person(name: str) -> AggregatedPerson:
    det = (Detection(text=name, start=0, end=len(name), page_index=0,
                      detector=DetectorName.SPACY, confidence=0.5,
                      metadata={"pattern": "spacy_ner"}),)
    candidate = Candidate.new(name, name, 0, len(name), 0, det)
    state = CandidateState()
    state.final_confidence = 0.70
    state.decision = Decision.REVIEW
    from src.core.models import FeatureVector
    state.feature_vector = FeatureVector(
        token_count=len(name.split()), char_length=len(name), has_title=0, has_honorific=0,
        first_name_dict_hits=0, last_name_dict_hits=0, dict_hit_ratio=0.0, capitalized_ratio=1.0,
        initial_token_count=0, regex_confidence=0.0, dictionary_confidence=0.0, spacy_confidence=0.5,
        detector_count=1, preceding_word_is_stopword=0, following_char_is_punct=0, occurs_in_quotes=0,
    )
    cr = CandidateResult(candidate=candidate, state=state)
    return AggregatedPerson(normalized_text=name, display_text=name, mentions=[cr])


def test_feedback_store_logs_and_dedupes(tmp_path):
    store = FeedbackStore(tmp_path / "feedback")
    person = _fake_review_person("Roger Walsh")

    logged = store.log_review_persons([person], "some_file.txt")
    assert logged == 1

    # Logging the same person again must not create a duplicate
    logged_again = store.log_review_persons([person], "another_file.txt")
    assert logged_again == 0

    pending = store.load_pending()
    assert len(pending) == 1
    assert pending[0].text == "Roger Walsh"
    assert pending[0].status == "pending"


def test_feedback_store_confirm_workflow(tmp_path):
    store = FeedbackStore(tmp_path / "feedback")
    person = _fake_review_person("Deane Shapiro")
    store.log_review_persons([person], "some_file.txt")

    pending = store.load_pending()
    record = pending[0]
    record.status = "confirmed_person"
    store.append_confirmed(record)

    confirmed = store.load_confirmed()
    assert len(confirmed) == 1
    assert confirmed[0].status == "confirmed_person"
    assert len(confirmed[0].features) == 16


def test_feedback_record_round_trips_through_json():
    record = FeedbackRecord(
        feedback_id="abc-123", source_file="f.txt", text="Jane Doe",
        normalized_text="Jane Doe", location="line 1, col 1", confidence=0.7,
        features=[1.0] * 16, logged_at_utc="2026-01-01T00:00:00Z",
    )
    restored = FeedbackRecord.from_dict(record.as_dict())
    assert restored.text == "Jane Doe"
    assert restored.features == record.features
