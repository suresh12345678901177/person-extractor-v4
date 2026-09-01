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
