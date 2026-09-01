"""Unit tests for src.candidate.boundary_refiner - the within-document
frequency-based trim, the unconditional stopword-edge trim, and the
batch-level combined-count refinement added this session (see
project memory / commit history for the "James Nick Today" /
"Devraj Bhatt We" bugs these fix)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.candidate.boundary_refiner import (
    refine_boundaries,
    refine_combined_counts,
    trim_stopword_edges,
)
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.line_indexer import LineIndex

KB = KnowledgeBase.load(BASE_DIR / "assets")


class _DocBuilder:
    """Builds a document out of pieces while tracking exact offsets, so
    test fixtures never rely on hand-counted character positions (the
    same silent-error trap scripts/build_benchmark_from_markup.py exists
    to avoid) - important here specifically because pieces like "Severus
    Morris" and "Testing Severus Morris" are substrings of each other,
    which would make after-the-fact index()-based lookup ambiguous."""

    def __init__(self) -> None:
        self.text = ""
        self.candidates: list[CandidateResult] = []

    def plain(self, s: str) -> "_DocBuilder":
        self.text += s
        return self

    def name(self, s: str) -> "_DocBuilder":
        start = len(self.text)
        self.text += s
        end = len(self.text)
        detection = Detection(
            text=s, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55, metadata={"pattern": "bare"},
        )
        candidate = Candidate.new(s, s, start, end, 0, (detection,))
        self.candidates.append(CandidateResult(candidate=candidate))
        return self


def test_refine_boundaries_trims_rare_noisy_variant_to_dominant_core():
    b = _DocBuilder()
    b.name("Severus Morris").plain(" said hi. Testing ").name("Testing Severus Morris")
    b.plain(" here. ").name("Severus Morris").plain(" again. ").name("Severus Morris").plain(" once more.")
    line_index = LineIndex.build(b.text)

    refined = refine_boundaries(b.candidates, line_index)

    texts = [c.candidate.normalized_text for c in refined]
    assert texts.count("Severus Morris") == 4
    assert "Testing Severus Morris" not in texts

    trimmed = refined[1]
    assert trimmed.candidate.text == "Severus Morris"
    assert b.text[trimmed.candidate.start:trimmed.candidate.end] == "Severus Morris"


def test_refine_boundaries_does_not_trim_when_core_is_not_established():
    # Only ONE clean occurrence - below MIN_CORE_OCCURRENCES=3, so the
    # noisy variant must be left alone (this is exactly the gap
    # trim_stopword_edges exists to cover for the stopword case, but for
    # a non-stopword glue word like "Testing" there is no safe fallback -
    # the candidate should simply pass through unchanged).
    b = _DocBuilder()
    b.name("Devraj Bhatt").plain(" created the group. ").name("Testing Devraj Bhatt").plain(" here.")
    line_index = LineIndex.build(b.text)

    refined = refine_boundaries(b.candidates, line_index)

    texts = [c.candidate.normalized_text for c in refined]
    assert texts == ["Devraj Bhatt", "Testing Devraj Bhatt"]


def test_trim_stopword_edges_trims_trailing_stopword():
    b = _DocBuilder()
    b.plain("Yesterday ").name("Devraj Bhatt Will").plain(" do it.")
    line_index = LineIndex.build(b.text)

    refined = trim_stopword_edges(b.candidates, KB, line_index)

    assert refined[0].candidate.normalized_text == "Devraj Bhatt"
    assert b.text[refined[0].candidate.start:refined[0].candidate.end] == "Devraj Bhatt"


def test_trim_stopword_edges_trims_leading_stopword():
    b = _DocBuilder()
    b.name("We Devraj Bhatt").plain(" confirmed the order.")
    line_index = LineIndex.build(b.text)

    refined = trim_stopword_edges(b.candidates, KB, line_index)

    assert refined[0].candidate.normalized_text == "Devraj Bhatt"


def test_trim_stopword_edges_trims_both_edges_in_one_pass():
    b = _DocBuilder()
    b.name("We Devraj Bhatt Will").plain(" do it.")
    line_index = LineIndex.build(b.text)

    refined = trim_stopword_edges(b.candidates, KB, line_index)

    assert refined[0].candidate.normalized_text == "Devraj Bhatt"


def test_trim_stopword_edges_leaves_clean_candidate_unchanged():
    b = _DocBuilder()
    b.name("Devraj Bhatt").plain(" confirmed the order.")
    line_index = LineIndex.build(b.text)

    refined = trim_stopword_edges(b.candidates, KB, line_index)

    assert refined[0].candidate.normalized_text == "Devraj Bhatt"
    assert refined[0] is b.candidates[0]


def test_trim_stopword_edges_never_trims_below_one_token():
    b = _DocBuilder()
    b.name("We Will").plain(" confirm it.")
    line_index = LineIndex.build(b.text)

    refined = trim_stopword_edges(b.candidates, KB, line_index)

    # both tokens are stopwords - trimming stops once a single token
    # remains, it never produces an empty candidate.
    assert refined[0].candidate.normalized_text in ("We", "Will")


def test_refine_combined_counts_merges_noise_into_dominant_core():
    combined = {
        "Severus Morris": {"display": "Severus Morris", "occurrences": 80, "files": ["a.txt"]},
        "Testing Severus Morris": {"display": "Testing Severus Morris", "occurrences": 2, "files": ["b.txt"]},
    }

    merged = refine_combined_counts(combined)

    assert "Testing Severus Morris" not in merged
    assert merged["Severus Morris"]["occurrences"] == 82
    assert set(merged["Severus Morris"]["files"]) == {"a.txt", "b.txt"}


def test_refine_combined_counts_leaves_unrelated_entries_alone():
    combined = {
        "Severus Morris": {"display": "Severus Morris", "occurrences": 80, "files": ["a.txt"]},
        "Andrew Morris": {"display": "Andrew Morris", "occurrences": 10, "files": ["a.txt"]},
    }

    merged = refine_combined_counts(combined)

    assert set(merged.keys()) == {"Severus Morris", "Andrew Morris"}
    assert merged["Andrew Morris"]["occurrences"] == 10


def test_refine_combined_counts_does_not_merge_below_dominance_threshold():
    combined = {
        "Severus Morris": {"display": "Severus Morris", "occurrences": 5, "files": ["a.txt"]},
        "Testing Severus Morris": {"display": "Testing Severus Morris", "occurrences": 2, "files": ["b.txt"]},
    }

    merged = refine_combined_counts(combined)

    # core_count(5) < own_count(2) * MIN_DOMINANCE_RATIO(3) == 6, so this
    # must NOT merge.
    assert "Testing Severus Morris" in merged
    assert merged["Severus Morris"]["occurrences"] == 5
