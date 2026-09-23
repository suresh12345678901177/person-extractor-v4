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
        preceding_context_is_person_cue=0, following_context_is_nonperson_cue=0,
        is_isolated_line=0,
    )
    cr = CandidateResult(candidate=candidate, state=state)
    return AggregatedPerson(normalized_text=name, display_text=name, mentions=[cr])


def test_feedback_store_logs_and_dedupes(tmp_path):
    store = FeedbackStore(tmp_path / "feedback")
    person = _fake_review_person("Roger Walsh")

    logged = store.log_review_persons([person], "some_file.txt", "Roger Walsh")
    assert logged == 1

    # Logging the same person again must not create a duplicate
    logged_again = store.log_review_persons([person], "another_file.txt", "Roger Walsh")
    assert logged_again == 0

    pending = store.load_pending()
    assert len(pending) == 1
    assert pending[0].text == "Roger Walsh"
    assert pending[0].status == "pending"
    assert pending[0].context_text == "Roger Walsh"


def test_existing_texts_cache_stays_correct_across_many_calls(tmp_path):
    # Regression test for the 2026-09-22 feedback_logging speed fix:
    # _existing_texts() is now cached on the FeedbackStore instance
    # instead of re-reading both JSONL files from disk on every call
    # (measured ~1.6-1.8s per file in a real batch scan, larger than
    # every other pipeline stage combined). This must not change
    # dedup correctness across many calls within one instance's
    # lifetime - the same shape as a real multi-file batch scan.
    store = FeedbackStore(tmp_path / "feedback")

    for i in range(5):
        person = _fake_review_person(f"Person{i} Test")
        logged = store.log_review_persons([person], f"file_{i}.txt", f"Person{i} Test")
        assert logged == 1, f"iteration {i}: first sighting must log 1 new record"

    # Re-logging every earlier person again (as a later file in the same
    # "batch") must dedupe against ALL of them, not just the most recent -
    # proves the cache accumulates correctly, not just remembers the last call.
    for i in range(5):
        person = _fake_review_person(f"Person{i} Test")
        logged_again = store.log_review_persons([person], "later_file.txt", f"Person{i} Test")
        assert logged_again == 0, f"iteration {i}: repeat sighting must not create a duplicate"

    pending = store.load_pending()
    assert len(pending) == 5
    assert store._existing_texts_cache is not None
    assert len(store._existing_texts_cache) == 5


def test_feedback_store_confirm_workflow(tmp_path):
    store = FeedbackStore(tmp_path / "feedback")
    person = _fake_review_person("Deane Shapiro")
    store.log_review_persons([person], "some_file.txt", "Deane Shapiro")

    pending = store.load_pending()
    record = pending[0]
    record.status = "confirmed_person"
    store.append_confirmed(record)

    confirmed = store.load_confirmed()
    assert len(confirmed) == 1
    assert confirmed[0].status == "confirmed_person"
    assert len(confirmed[0].features) == 19  # FeatureVector field count (2026-09-17: +3 context features)


def test_feedback_record_round_trips_through_json():
    record = FeedbackRecord(
        feedback_id="abc-123", source_file="f.txt", text="Jane Doe",
        normalized_text="Jane Doe", location="line 1, col 1", confidence=0.7,
        features=[1.0] * 16, logged_at_utc="2026-01-01T00:00:00Z",
        context_text="Jane Doe was seen near the warehouse.",
    )
    restored = FeedbackRecord.from_dict(record.as_dict())
    assert restored.text == "Jane Doe"
    assert restored.features == record.features
    assert restored.context_text == "Jane Doe was seen near the warehouse."


def test_feedback_record_from_dict_defaults_context_text_for_legacy_rows():
    # Records logged before context_text existed have no such key in their
    # stored dict at all - from_dict must still deserialize them cleanly.
    legacy_dict = {
        "feedback_id": "abc-123", "source_file": "f.txt", "text": "Jane Doe",
        "normalized_text": "Jane Doe", "location": "line 1, col 1", "confidence": 0.7,
        "features": [1.0] * 16, "logged_at_utc": "2026-01-01T00:00:00Z",
        "status": "pending", "labeled_at_utc": None,
    }
    restored = FeedbackRecord.from_dict(legacy_dict)
    assert restored.context_text == ""
