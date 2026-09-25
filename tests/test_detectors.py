"""Unit tests for the detection layer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.detection.dictionary_detector import DictionaryDetector
from src.detection.regex_detector import RegexDetector
from src.knowledge.knowledge_base import KnowledgeBase

KB = KnowledgeBase.load(BASE_DIR / "assets")


def test_regex_detector_finds_titled_name():
    detections = RegexDetector().detect("Dr. Suresh Kumar attended the meeting.", 0)
    assert any("Suresh Kumar" in d.text for d in detections)
    titled = [d for d in detections if d.metadata.get("pattern") == "titled"]
    assert titled
    assert titled[0].confidence == RegexDetector.TITLED_CONFIDENCE


def test_regex_detector_finds_bare_name():
    detections = RegexDetector().detect("John Smith walked into the room.", 0)
    assert any(d.text == "John Smith" for d in detections)


def test_regex_detector_never_spans_a_newline():
    detections = RegexDetector().detect("Suresh\nJones", 0)
    assert not any("\n" in d.text for d in detections)
    assert not any(d.text == "Suresh Jones" for d in detections)


def test_regex_detector_empty_text_returns_empty_list():
    assert RegexDetector().detect("", 0) == []


def test_dictionary_detector_finds_known_name_pair():
    detections = DictionaryDetector(KB).detect("Amit Sharma joined the company today.", 0)
    assert any(d.text == "Amit Sharma" for d in detections)


def test_dictionary_detector_ignores_unknown_tokens():
    detections = DictionaryDetector(KB).detect("Xzqlt Vbnmp arrived late.", 0)
    assert detections == []


def test_dictionary_detector_finds_single_known_first_name_at_low_confidence():
    detections = DictionaryDetector(KB).detect("Suresh claimed he heard shouting.", 0)
    single_token = [d for d in detections if d.text == "Suresh"]
    assert single_token
    assert single_token[0].metadata.get("pattern") == "dictionary_single_token"
    assert single_token[0].confidence < DictionaryDetector.MIN_CONFIDENCE


def test_dictionary_detector_handles_possessive_suffix():
    detections = DictionaryDetector(KB).detect("similar to Geetha's old notebook.", 0)
    matches = [d for d in detections if d.text == "Geetha"]
    assert matches, f"expected a 'Geetha' detection, got: {[d.text for d in detections]}"
    assert matches[0].end - matches[0].start == len("Geetha")


def test_dictionary_detector_reads_whole_accented_words():
    """The old ASCII-only token pattern matched only the ASCII prefix of
    an accented word - 'François' produced a hit on the fragment 'Fran'."""
    detections = DictionaryDetector(KB).detect("Yesterday François Dubois called.", page_index=0)
    assert [d.text for d in detections] == ["François Dubois"]
    assert not any(d.text == "Fran" for d in detections)


def test_regex_detector_accepts_extended_latin_capitals():
    """Ł/Š/Ș capitals were outside the old hand-typed first-letter class."""
    texts = {d.text for d in RegexDetector().detect("we saw Łukasz Kowalski and Ștefan Popescu.", page_index=0)}
    assert {"Łukasz Kowalski", "Ștefan Popescu"} <= texts


def test_accent_variant_fallback_applies_only_to_non_ascii_tokens():
    assert KB.is_known_first_name("Ștefan")      # listed under another accent form
    assert KB.is_known_last_name("Nguyễn")       # listed as plain "nguyen"
    # Plain-ASCII text is never matched against accented-only entries:
    # "Francois" is listed only as "françois" among first names.
    assert KB.is_known_first_name("François")
    assert not KB.is_known_first_name("Francois")


def test_dictionary_detector_never_matches_inside_code_identifiers():
    """Regression (2026-09-25): 'Handler' - a listed surname - was matched
    inside clickHandler / errorHandler and ACCEPTED 146 times in a real scan."""
    text = "el.clickHandler(); on_Handler = errorHandler; getKumarData(); Handler2; x.Kumar_id"
    assert DictionaryDetector(KB).detect(text, 0) == []


def test_dictionary_detector_still_finds_whole_words_next_to_punctuation():
    # Control: punctuation never blocks a match, and neither does a digit
    # BEFORE the name - reference lists glue the note number on ("1Berger, A.").
    texts = {d.text for d in DictionaryDetector(KB).detect(
        "(Kumar) Dr.Suresh; O'Brien, McDonald met Handler.\n1Berger, A., Instabilities.", 0)}
    assert {"Kumar", "Suresh", "O'Brien", "McDonald", "Handler", "Berger"} <= texts, texts
