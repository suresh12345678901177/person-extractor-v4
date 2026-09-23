"""Unit tests for src.preprocessing.segmenter, particularly
line_containing()'s max_chars bound.

Written after a real audit (2026-09-23) found line_containing() had no
size cap at all: a document with no newline anywhere near a candidate's
offset (a minified JS bundle, a book dump with no paragraph breaks in
that region, a chat export whose "line" spans the whole conversation)
returned the ENTIRE remaining document as "context" - confirmed to have
produced up to 927,834-char single-record captures in
datasets/feedback/confirmed_labels.jsonl, including one real case's full
chat conversation for what should have been a short local snippet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.segmenter import line_containing


def test_short_line_is_returned_unmodified():
    text = "Suresh Kumar attended the meeting.\nHe left early."
    assert line_containing(text, 0) == "Suresh Kumar attended the meeting."


def test_offset_mid_document_returns_only_its_own_line():
    text = "First line here.\nSuresh Kumar attended.\nThird line here."
    offset = text.index("Suresh")
    assert line_containing(text, offset) == "Suresh Kumar attended."


def test_document_with_no_newlines_is_capped_not_returned_whole():
    """The exact real-world shape that caused the bug: a single-line
    (or no-newline-nearby) document of arbitrary size."""
    huge_doc = "x" * 500_000 + "Suresh" + "y" * 500_000
    offset = huge_doc.index("Suresh")

    result = line_containing(huge_doc, offset, max_chars=2000)

    assert len(result) <= 2000
    assert "Suresh" in result


def test_candidate_near_the_end_of_a_huge_line_is_still_included():
    """A naive "truncate from window_start" fix would cut off a
    candidate sitting near the end of a huge line - the window must be
    centered on offset, not anchored at the line's start."""
    huge_doc = "x" * 1_000_000 + "Suresh"
    offset = huge_doc.index("Suresh")

    result = line_containing(huge_doc, offset, max_chars=2000)

    assert "Suresh" in result
    assert len(result) <= 2000


def test_max_chars_default_is_generous_for_realistic_lines():
    """A realistic long-ish line (a few hundred chars, as seen in the
    project's real feedback data) is never truncated by the default."""
    line = "A" * 1500
    text = f"before\n{line}\nafter"
    offset = text.index(line)
    assert line_containing(text, offset) == line


def test_line_exactly_at_max_chars_boundary_is_untouched():
    text = "x" * 2000
    assert line_containing(text, 0, max_chars=2000) == text
