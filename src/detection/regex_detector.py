"""
src.detection.regex_detector
==============================
Regex-based person-name detector.

Two complementary patterns:
1. Titled names: an honorific/title followed by 1-4 capitalized tokens
   ("Dr. Suresh Kumar", "Mrs. Jane A. Doe").
2. Bare capitalized sequences: 2-4 consecutive capitalized tokens with no
   title ("John Smith"). Lower base confidence since bare capitalization
   alone is a weaker signal (also matches place/org names) - the
   Validation Firewall is responsible for filtering those, not the regex
   detector itself.

Both patterns use [ \\t] (never \\s) between tokens so a match can NEVER
span a newline - i.e. never merge two different lines/rows into one fake
"name". (This was a real bug found and fixed during V3 adversarial
testing: \\s+ let a match bleed across a table row boundary.) The
whitespace run between tokens is further capped at 1-2 characters (see
_SAFE_SEP) so a match also can't bridge 3+ spaces of deliberate table-
column padding on the SAME line, which \\t-vs-\\s alone doesn't prevent.

The titled pattern also allows STACKED titles ("Prof. Dr. Lalita Kher")
to all be consumed as the title prefix, not just the first one - without
that, the second title word ("Dr.") looks exactly like a valid name
token and gets wrongly captured as if it were the person's actual name,
splitting the real name off as a separate, unlabeled bare match.

A trailing possessive ('s / 's) is stripped from the matched span before
it becomes a Detection, same as dictionary_detector.py already does for
its own matches. Found via real testing: without this, "Wrenna
Solkiewicz's number changed" produced a SEPARATE candidate normalized to
"Wrenna Solkiewicz's" - a different aggregation key than plain "Wrenna
Solkiewicz" - fragmenting one real person's mentions into two weaker,
independently-scored report entries instead of contributing to the same
occurrence count.
"""

from __future__ import annotations

import re

from src.core.models import Detection, DetectorName
from src.detection.base_detector import BaseDetector
from src.detection.dictionary_detector import _strip_possessive
from src.preprocessing.latin import LATIN_LETTERS, LATIN_UPPER

# Token: capitalized word, allows hyphens/apostrophes (O'Brien, Smith-Jones).
# The accented-letter ranges (Latin-1 Supplement + Latin Extended-A) exist
# because the plain-ASCII-only version of this pattern silently couldn't
# match accented Latin names at all - not "match them oddly", literally
# zero detections, since a single non-ASCII character (e.g. the ç in
# "Francois") breaks the token match outright and the whole name vanishes
# with no trace anywhere in the pipeline. Deliberately still ASCII-first
# in the class (most real-world names are pure ASCII) with the accented
# ranges appended, not a full \w-based Unicode rewrite - keeps the
# character set to Latin-script names specifically, consistent with
# LanguageValidator's non-Latin-script hard reject elsewhere in this
# pipeline.
#
# The letter classes now come from src/preprocessing/latin.py, shared with
# StructureValidator/DictionaryDetector - which had stayed ASCII-only and
# were rejecting/fragmenting every accented name this detector found. The
# old hand-typed first-letter class [A-ZÀ-ÖØ-Þ] also missed Ł/Š/Č/Ž and
# Romanian/Vietnamese capitals (see latin.py's docstring).
_LATIN_LETTER = LATIN_LETTERS
_NAME_TOKEN = rf"[{LATIN_UPPER}][{_LATIN_LETTER}'\-]*\.?"

_TITLES = (
    r"Dr|Mr|Mrs|Ms|Miss|Prof|Professor|Rev|Fr|Sir|Madam|Capt|Col|"
    r"Gen|Lt|Sgt|Hon|Judge|Sen|Rep|Amb"
)

# A single non-title token, used to build the separator: two name tokens
# may be joined by [ \t]{1,2} ONLY if the whitespace is not immediately
# preceded by a genuine sentence-ending period/!/? - the match must
# never cross a real sentence boundary (same principle that already
# prevents crossing a newline). The one deliberate exception: a period
# directly after a SINGLE capital letter ("J.", "K.") is a mid-name
# initial, not a sentence end, and must still be allowed to join to the
# next token ("Cortland J. Dahl" must not be truncated to "Cortland J").
#
# Capped at 1-2 whitespace characters (not unbounded [ \t]+) so a match
# can tolerate one accidental extra space (real chat data really does
# have this - e.g. "Donald  Richard" with a stray double space) without
# also being able to bridge 3+ spaces of deliberate column padding in
# tabular/report-style text ("Kavya Bhandari    Design       Active").
# 3+ spaces between words is essentially never natural prose spacing;
# treating it as a hard boundary (same principle as never crossing a
# newline) stops a name from swallowing adjacent table-column values.
_SAFE_SEP = rf"(?:(?<=\b[{LATIN_UPPER}]\.)[ \t]{{1,2}}|(?<![.!?])[ \t]{{1,2}})"

_TITLED_NAME_RE = re.compile(
    rf"\b(?:{_TITLES})\.?[ \t]{{1,2}}(?:(?:{_TITLES})\.?[ \t]{{1,2}})*"
    rf"(?:{_NAME_TOKEN}{_SAFE_SEP}){{0,3}}{_NAME_TOKEN}"
)

_BARE_NAME_RE = re.compile(
    rf"\b{_NAME_TOKEN}(?:{_SAFE_SEP}{_NAME_TOKEN}){{1,3}}\b"
)


class RegexDetector(BaseDetector):
    name = DetectorName.REGEX

    TITLED_CONFIDENCE = 0.90
    BARE_CONFIDENCE = 0.55

    def detect(self, text: str, page_index: int) -> list[Detection]:
        if not text:
            return []

        detections: list[Detection] = []
        claimed_spans: list[tuple[int, int]] = []

        for match in _TITLED_NAME_RE.finditer(text):
            raw = match.group().strip()
            stemmed, trim = _strip_possessive(raw)
            detections.append(
                Detection(
                    text=stemmed,
                    start=match.start(),
                    end=match.end() - trim,
                    page_index=page_index,
                    detector=self.name,
                    confidence=self.TITLED_CONFIDENCE,
                    metadata={"pattern": "titled"},
                )
            )
            claimed_spans.append((match.start(), match.end()))

        for match in _BARE_NAME_RE.finditer(text):
            span = (match.start(), match.end())
            if self._overlaps(span, claimed_spans):
                continue  # already covered by a stronger titled match
            raw = match.group().strip()
            stemmed, trim = _strip_possessive(raw)
            detections.append(
                Detection(
                    text=stemmed,
                    start=match.start(),
                    end=match.end() - trim,
                    page_index=page_index,
                    detector=self.name,
                    confidence=self.BARE_CONFIDENCE,
                    metadata={"pattern": "bare"},
                )
            )

        return detections

    @staticmethod
    def _overlaps(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
        start, end = span
        return any(not (end <= o_start or start >= o_end) for o_start, o_end in others)
