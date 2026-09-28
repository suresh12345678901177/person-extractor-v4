"""Tests for canonical-span selection in src/candidate/candidate_factory.py -
specifically the dictionary-single-token boundary rescue (2026-09-24)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.candidate.candidate_factory import _select_canonical_span
from src.core.models import Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _group(text: str, dictionary_word: str) -> list[Detection]:
    start = text.index(dictionary_word)
    return [
        Detection(dictionary_word, start, start + len(dictionary_word), 0, DetectorName.DICTIONARY, 0.4,
                  {"pattern": "dictionary_single_token"}),
        Detection(text, 0, len(text), 0, DetectorName.REGEX, 0.55, {"pattern": "bare"}),
    ]


def test_one_word_dictionary_name_widens_to_unknown_surname():
    """'Nadia' used to win boundary selection over 'Nadia Kovalenko',
    truncating the name (10.5% of accepted benchmark mentions were cut
    short like this)."""
    assert _select_canonical_span(_group("Nadia Kovalenko", "Nadia"), KB).text == "Nadia Kovalenko"
    assert _select_canonical_span(_group("Devraj Bhatt", "Bhatt"), KB).text == "Devraj Bhatt"


def test_no_widening_onto_ordinary_words():
    assert _select_canonical_span(_group("Thanks Nadia", "Nadia"), KB).text == "Nadia"
    assert _select_canonical_span(_group("Nadia Report", "Nadia"), KB).text == "Nadia"


def test_without_a_knowledge_base_previous_behavior_is_kept():
    assert _select_canonical_span(_group("Nadia Kovalenko", "Nadia")).text == "Nadia"


def test_one_word_dictionary_name_widens_to_a_middle_initial_name():
    """Regression (2026-09-28): 'Kurt' won over 'Kurt D. DelBene' because the
    widening only ever considered two-token spans - the surname was lost."""
    assert _select_canonical_span(_group("Kurt D. DelBene", "Kurt"), KB).text == "Kurt D. DelBene"
    # Control: the token after the initial must itself read as a name.
    assert _select_canonical_span(_group("Kurt D. Report", "Kurt"), KB).text == "Kurt"


def test_middle_initial_shape():
    from src.candidate.candidate_factory import is_middle_initial_name
    assert is_middle_initial_name("Craig J. Mundie") and is_middle_initial_name("Mary A. B. Jones")
    assert not is_middle_initial_name("Craig Mundie")      # no initial
    assert not is_middle_initial_name("J. R. Smith")        # starts with an initial
    assert not is_middle_initial_name("Craig J Mundie")     # initial without its period
