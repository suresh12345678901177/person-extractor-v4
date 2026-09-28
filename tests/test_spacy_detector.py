"""Tests for SpacyDetector's line-bounded chunking (the fix for spaCy's
1,000,000-char nlp.max_length ceiling silently dropping all detections on
large documents) and its offset correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.detection.spacy_detector import iter_line_bounded_chunks


def _reconstruct(text: str, max_chars: int) -> str:
    return "".join(chunk for _, chunk in iter_line_bounded_chunks(text, max_chars))


def _offsets_correct(text: str, max_chars: int) -> bool:
    for offset, chunk in iter_line_bounded_chunks(text, max_chars):
        if text[offset:offset + len(chunk)] != chunk:
            return False
    return True


def test_short_text_yields_single_unmodified_chunk():
    text = "John Smith met Sarah Connor.\nSecond line here.\n"
    chunks = list(iter_line_bounded_chunks(text, max_chars=1000))
    assert chunks == [(0, text)]


def test_chunks_never_exceed_max_chars_and_split_only_on_lines():
    lines = [f"line {i} some name-shaped text here\n" for i in range(500)]
    text = "".join(lines)
    max_chars = 500  # forces many chunk boundaries

    chunks = list(iter_line_bounded_chunks(text, max_chars))
    assert len(chunks) > 1
    for _, chunk in chunks:
        assert len(chunk) <= max_chars
        # every chunk boundary falls on a line boundary: each chunk
        # either ends with a newline or is the final chunk
    assert _reconstruct(text, max_chars) == text
    assert _offsets_correct(text, max_chars)


def test_single_line_longer_than_max_chars_is_hard_split():
    # simulates a pathological one-line JSON/hex blob some Android dumps
    # contain - can't be kept in one spaCy call either way
    text = "x" * 1000 + "\n" + "short line\n"
    max_chars = 300

    chunks = list(iter_line_bounded_chunks(text, max_chars))
    for _, chunk in chunks:
        assert len(chunk) <= max_chars
    assert _reconstruct(text, max_chars) == text
    assert _offsets_correct(text, max_chars)


def test_exotic_unicode_line_separators_still_reconstruct_correctly():
    # NEL (U+0085), LS (U+2028), PS (U+2029) are all real line-break
    # characters as far as splitlines(keepends=True) is concerned - make
    # sure chunking still reconstructs exactly and offsets stay correct.
    text = "Alpha Beta Gamma DeltaEpsilon Zeta " + ("filler " * 100)
    max_chars = 40

    assert _reconstruct(text, max_chars) == text
    assert _offsets_correct(text, max_chars)


def test_boundary_exact_multiple_of_max_chars():
    text = ("abcde\n" * 10)  # 60 chars total, lines of 6 chars each
    max_chars = 30  # exact multiple

    chunks = list(iter_line_bounded_chunks(text, max_chars))
    assert _reconstruct(text, max_chars) == text
    assert _offsets_correct(text, max_chars)
    for _, chunk in chunks:
        assert len(chunk) <= max_chars


@pytest.mark.parametrize("max_chars", [1, 2, 5, 13, 37, 1000])
def test_various_chunk_sizes_always_reconstruct(max_chars):
    text = "Marcus Webb spoke with Dana Okafor.\nThey discussed the deal.\n\nRegards,\nTeam\n"
    assert _reconstruct(text, max_chars) == text
    assert _offsets_correct(text, max_chars)


def test_spacy_detector_finds_person_across_a_forced_chunk_boundary():
    """End-to-end: with a tiny MAX_CHUNK_CHARS forcing many chunks, a
    real name sitting near a chunk boundary must still be found at the
    correct absolute offset in the original text - proves the offset
    math survives the chunk -> global-position translation, not just the
    chunking helper in isolation."""
    from src.detection.spacy_detector import SpacyDetector

    detector = SpacyDetector()
    original_max = SpacyDetector.MAX_CHUNK_CHARS
    SpacyDetector.MAX_CHUNK_CHARS = 60  # force multiple small chunks
    try:
        padding = "The weather today is quite fine and mild for this time of year.\n" * 3
        text = padding + "According to Marcus Webb, the timeline needs to change.\n" + padding
        detections = detector.detect(text, page_index=0)
    finally:
        SpacyDetector.MAX_CHUNK_CHARS = original_max

    person_texts = {d.text for d in detections}
    assert any("Marcus Webb" in t or "Webb" in t for t in person_texts), person_texts

    for d in detections:
        assert text[d.start:d.end] == d.text


def test_parallel_detection_matches_sequential_exactly():
    """n_process > 1 (config detection.spacy_processes) must return the
    exact same detections, in the same order, as the sequential path -
    it only spreads the same chunks over worker processes."""
    from src.detection.spacy_detector import SpacyDetector

    original_max = SpacyDetector.MAX_CHUNK_CHARS
    SpacyDetector.MAX_CHUNK_CHARS = 120  # force several chunks
    try:
        text = (
            "According to Marcus Webb, the timeline needs to change.\n"
            "Sarah Connor replied that Priya Raman would sign off.\n"
            "The weather today is quite fine and mild for this time of year.\n"
        ) * 4
        sequential = SpacyDetector(n_process=1).detect(text, page_index=0)
        parallel = SpacyDetector(n_process=2).detect(text, page_index=0)
    finally:
        SpacyDetector.MAX_CHUNK_CHARS = original_max

    key = lambda ds: [(d.text, d.start, d.end, d.confidence) for d in ds]
    assert sequential, "expected at least one PERSON detection"
    assert key(parallel) == key(sequential)


def test_configured_spacy_model_is_the_one_loaded():
    """detection.spacy_model (config.py) must reach spacy.load - the
    model_name argument used to be accepted and silently ignored. The
    default stays en_core_web_sm."""
    import copy

    from config import DEFAULT_CONFIG
    from src.detection.detector_manager import DetectorManager
    from src.detection.spacy_detector import SpacyDetector
    from src.knowledge.knowledge_base import KnowledgeBase

    assert DEFAULT_CONFIG["detection"]["spacy_model"] == "en_core_web_sm"
    assert SpacyDetector().model_version.startswith("en_core_web_sm-")

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["detection"]["spacy_model"] = "en_core_web_sm"
    manager = DetectorManager(config, KnowledgeBase(assets_dir=Path(".")))
    spacy_detector = next(d for d in manager.detectors if isinstance(d, SpacyDetector))
    assert spacy_detector.model_name == "en_core_web_sm"


def test_missing_spacy_model_names_the_download_command():
    from src.detection.spacy_detector import SpacyDetector

    with pytest.raises(RuntimeError, match="spacy download en_core_web_nonexistent"):
        SpacyDetector(model_name="en_core_web_nonexistent")


def test_name_segments_split_at_line_breaks_and_digit_tokens_only():
    from src.detection.spacy_detector import SpacyDetector

    def segs(text):
        return [text[s:e] for s, e in SpacyDetector._name_segments(text)]

    assert segs("Marcus  Webb") == ["Marcus  Webb"]                       # unchanged, spacing kept
    assert segs("Marco Silvestri\nDragan Kovac") == ["Marco Silvestri", "Dragan Kovac"]
    assert segs("Farhan Vora 919812345671 Meera Iyengar Hi") == ["Farhan Vora", "Meera Iyengar Hi"]
    assert segs("+91-98123 Rohan") == ["Rohan"]
    assert segs("A2B 42") == []


def test_one_line_call_log_keeps_both_names_as_candidates(tmp_path):
    """Regression (2026-09-28, benchmark_003): spaCy tagged 'Farhan Vora
    919812345671 Meera Iyengar Hi' as one person, and candidate merging then
    kept only 'Meera Iyengar' - 'Farhan Vora' at that spot was lost."""
    import copy
    from config import BASE_DIR, DEFAULT_CONFIG
    from src.pipeline.orchestrator import Pipeline

    text = ("919812345670 Farhan Vora 919812345671 Meera Iyengar Hi 919812345670 "
            "Where's the vendor list Farhan Vora 919812345671 Sterling Finch Logistics sent it already\n")
    sample = tmp_path / "call_log.txt"
    sample.write_text(text, encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    result = Pipeline(config=config, base_dir=BASE_DIR).run(sample)

    mentions = [m for p in [*result.persons, *result.review_persons] for m in p.mentions]
    first = text.index("Farhan Vora")
    assert any(m.candidate.start == first and m.candidate.text == "Farhan Vora" for m in mentions), \
        [(m.candidate.start, m.candidate.text) for m in mentions]
