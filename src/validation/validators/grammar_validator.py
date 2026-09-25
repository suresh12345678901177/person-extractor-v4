"""
src.validation.validators.grammar_validator
==============================================
Targets the classic NER false-positive pattern: a normal English word
that happens to sit at the very start of a sentence gets capitalized by
ordinary orthography rules, and a statistical NER model (spaCy) or a
weak single-token dictionary hit occasionally mislabels it as a person.

Rule: hard-reject a candidate ONLY if ALL of the following hold:
  - it is exactly one token,
  - it has NO dictionary-detector evidence (single-token dictionary
    matches are exempt - a name confirmed by the offline dictionary at
    sentence-start, e.g. "Suresh claimed...", is real signal, not noise),
  - it has NO titled-pattern evidence,
  - its span starts a new sentence or paragraph (immediately after
    '.', '!', '?', or the very start of the document/paragraph).

This is intentionally narrow: it only fires on the highest-risk
combination (single token + zero dictionary support + sentence-initial),
so it cannot demote a multi-token name or anything the knowledge base
already confirms - keeping recall intact while closing a real,
observed false-positive path (seen in testing: spaCy occasionally
tags an ordinary sentence-initial word as PERSON).
"""

from __future__ import annotations

import re

from src.candidate.name_propagation import DOCUMENT_NAME_PATTERN
from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

_SENTENCE_END_RE = re.compile(r'[.!?]["\'\u2019\u201d]?\s*$')

# _is_sentence_initial only needs to see the last real (non-whitespace/
# quote) character before `start` - this bounds the lookback slice
# instead of copying document_text[:start] in full. That full-prefix
# slice is O(start) per candidate; on a large document with many
# single-token candidates scattered throughout, that is O(candidates x
# document_length) overall - confirmed directly on a real 13.7MB case
# file (53,587 candidates) where this line alone made validation hang
# for over an hour. 200 chars is far more than any real run of
# whitespace/opening-quote characters before real content; the rare
# pathological case (a 200+ char run of only whitespace/quotes) falls
# back to the full slice, which is correct, just no longer the common
# path's cost.
_LOOKBACK_WINDOW = 200


def _is_sentence_initial(document_text: str, start: int) -> bool:
    """True if `start` begins a new sentence/paragraph: nothing but
    whitespace/opening-quote characters between it and either the start
    of the text or the nearest preceding sentence-ending punctuation -
    OR the candidate is the first real content on its own line. The
    line-start check matters specifically for chat-export data (found
    via real testing, case 930204601): a stray single-word reaction
    starting a brand-new chat message/turn ("Fresh", "Send", "Yoooo")
    is exactly as much a "fresh, unconfirmed context" as a new sentence
    is, but the prior line in messy real chat text frequently doesn't
    end in real sentence punctuation at all (a phone number, a sender
    name, an emoji) - so without this, a new-line-starting single-token
    spaCy false positive sailed straight past this validator's safety
    net, the same one that already correctly catches an ordinary
    sentence-initial word."""
    if start == 0:
        return True

    line_start = document_text.rfind("\n", 0, start) + 1
    line_prefix = document_text[line_start:start]
    if line_prefix.strip(' \t"\'\u2018\u201c(') == "":
        return True  # nothing but leading whitespace/quotes on this line so far

    # Bounded lookback (not document_text[:start] in full - see
    # _LOOKBACK_WINDOW's docstring): only the last real character before
    # `start` matters, so a small fixed window is enough almost always.
    window_start = max(0, start - _LOOKBACK_WINDOW)
    before = document_text[window_start:start]
    # Walk back past whitespace and opening quote/bracket characters to
    # find the last "real" character before this candidate.
    stripped = before.rstrip(' \t\n"\'\u2018\u201c(')
    if not stripped:
        if window_start == 0:
            return True  # only whitespace/quotes before it -> start of text
        # Rare pathological case: 200+ consecutive whitespace/quote
        # characters right before this candidate - fall back to the
        # full prefix to stay correct (this branch is never the common
        # case that made the bounded window worth adding).
        before_full = document_text[:start]
        stripped_full = before_full.rstrip(' \t\n"\'\u2018\u201c(')
        if not stripped_full:
            return True
        return bool(_SENTENCE_END_RE.search(stripped_full + " "))

    return bool(_SENTENCE_END_RE.search(stripped + " "))


class GrammarValidator(BaseValidator):
    name = "grammar"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()
        if len(tokens) != 1:
            return ValidationResult(self.name, passed=True, score=0.0, message="Multi-token candidate, not applicable")

        detections = candidate.candidate.source_detections
        # document_name = the first name of a full name this document already
        # accepted (src/candidate/name_propagation.py) - dictionary-grade
        # evidence for this purpose ("Katya said..." at sentence start).
        has_dictionary_evidence = any(
            d.metadata.get("pattern", "").startswith("dictionary")
            or d.metadata.get("pattern") == DOCUMENT_NAME_PATTERN
            for d in detections
        )
        has_title_evidence = any(d.metadata.get("pattern") == "titled" for d in detections)

        if has_dictionary_evidence or has_title_evidence:
            return ValidationResult(self.name, passed=True, score=0.0, message="Dictionary/title-confirmed, exempt")

        if _is_sentence_initial(document_text, candidate.candidate.start):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Single token with no dictionary/title support at sentence-initial "
                        "position - likely ordinary capitalization, not a name",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="Not sentence-initial")
