"""Tests for src.preprocessing.language_filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.language_filter import MIN_WORDS_FOR_LANGUAGE_CHECK, check_english, is_short_non_prose

KB = KnowledgeBase.load(BASE_DIR / "assets")


def test_short_text_is_always_treated_as_english():
    assert check_english("0g ~~ | Suresh 8 ~ ## 4)", KB).is_english


def test_short_non_prose_catches_ocr_scraps_and_terse_records():
    # (2026-09-28, exp08) Made-up examples of the shapes found in a real case:
    # a garbled OCR scrap of an image, a lone word, a terse key/value record.
    assert is_short_non_prose("0g ~~ | Suresh 8 ~ ఇ అ ## 4)", KB)
    assert is_short_non_prose("Suresh", KB)
    assert is_short_non_prose("caseNo: 100, station: Central, officerName: Ravi Kumar", KB)


def test_short_genuine_notes_are_prose():
    for note in ("Call Rajesh tomorrow about the payment.",
                 "Suresh met Priya at the station and gave her the keys.",
                 "Thanks Nadia, see you soon"):
        assert not is_short_non_prose(note, KB), note


def test_long_text_is_never_short_non_prose():
    text = " ".join(["Suresh"] * MIN_WORDS_FOR_LANGUAGE_CHECK)
    assert not is_short_non_prose(text, KB)
