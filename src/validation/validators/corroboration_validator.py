"""
src.validation.validators.corroboration_validator
=====================================================
Hard-rejects a candidate whose ONLY evidence is a bare (untitled) regex
shape-match, with no dictionary hit, no titled pattern, and no spaCy
agreement. Found necessary via real-world testing on a bibliography/
reference-heavy document: book titles, publisher names, and journal
titles are all just consecutive capitalized words, indistinguishable
from a bare name by shape alone. On that document, ~900+ such spans
("Random House", "Buddhist Perspective", "Mouse Model") were correctly
kept out of ACCEPTED by the Decision Engine's knowledge-corroboration
gate, but still piled into REVIEW, making the tool practically unusable
despite being technically precision-safe.

This validator moves that same "needs at least one corroborating
signal" requirement earlier and stronger: from a soft decision-engine
cap to a hard, explained rejection. A bare regex match needs SOME other
detector or the offline dictionary to agree before it is worth a
human's attention at all.

Documented tradeoff: a genuine person name that is neither in the
knowledge base's dictionary NOR recognized by spaCy will now be
rejected outright rather than flagged for review. Given this project's
explicit, stated priority ("no false positives" over recall), this is
the correct default - see README "Known limitations".

A second corroborating signal, added after real-case testing (case
930204630): a name entirely missing from the dictionary AND rarely
tagged by spaCy in bare, context-free chat lines (a small NER model has
little to work with on a lone capitalized line with no sentence
structure around it - observed directly: spaCy caught a real recurring
name in only ~9 of its ~120 raw appearances in one such document) was
getting hard-rejected on the large majority of its real mentions,
despite being an obviously real, consistently-used name within that
one document. HIGH within-document repetition of the exact same bare
shape is itself meaningful evidence - the same "does this exact form
recur far more than chance in THIS document" principle boundary_refiner
already uses, applied here as a corroborating signal rather than a
boundary trim. Deliberately a HIGH bar (MIN_REPETITION_FOR_CORROBORATION),
not a low one: this validator's own history (see above) is that
low-frequency bare matches ARE the false-positive problem it exists to
prevent, so only very heavy, consistent repetition - far more than an
incidental phrase would show even in repetitive text - counts.

Threshold lowered 10 -> 4 (2026-08-19) after a controlled measurement:
built real names with EMPIRICALLY VERIFIED low spaCy hit rates (0-37%,
matching the real ~7.5% hit rate found on case 930204630's "Dragan
Kovac") at repetition counts 3/5/8/12, alongside distractor phrases at
matching counts. Result: a real name and a junk phrase at the SAME
repetition count flip REVIEW/rejected at the exact same threshold
boundary - the signal is symmetric, not biased toward false accepts.
Also found: names with 5+ occurrences almost always pick up SOME spaCy
support across their many mentions regardless of threshold, so the
threshold specifically only matters for the sparsest real misses (<=4
occurrences, e.g. "Nadia Kovalenko" at 4 in-file mentions) - exactly
the gap this lower bar targets, with no measured precision cost since
every distractor still capped at REVIEW, never ACCEPTED, unless it also
independently satisfied the ML classifier (a separate, later gate).

A third refinement, added after a deliberate stress test (2026-08-19):
spaCy alone is not always trustworthy corroboration. "Ruby Chamber" (a
fake storage-ledger label, not a person) got tagged PERSON by spaCy on
11 of its 12 appearances, purely because "Ruby" is a common first name
in spaCy's training data and primes it to expect a name to follow -
verified directly (spaCy's own POS tagger even agreed, tagging
"Chamber" as PROPN in that context - internally consistent in its own
mistake, so no cheap spaCy-internal signal catches this). For a
candidate whose ONLY evidence is spaCy, if its first token is a known
first-name/common-noun collision word (assets/ambiguous_words/
ambiguous_first_names.txt), spaCy alone is no longer enough - it falls
through to the same repetition-or-reject path a bare regex match would
use. This does not reject the word outright: a genuine person named
"Ruby Chamber" is still caught the same way any other name lacking
dictionary support is - by heavy in-document repetition or a
title/dictionary hit on either token.
"""

from __future__ import annotations

from src.core.models import CandidateResult, DetectorName, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.segmenter import count_occurrences
from src.validation.base_validator import BaseValidator


class CorroborationValidator(BaseValidator):
    name = "corroboration"

    MIN_REPETITION_FOR_CORROBORATION = 4

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        detections = candidate.candidate.source_detections
        text = candidate.candidate.normalized_text

        # Computed unconditionally, even on the branches below that don't
        # need it to decide pass/fail - DecisionEngine's ML-high-confidence
        # knowledge-corroboration path reuses this exact count later (see
        # CandidateState.repetition_count's docstring), so it must be
        # recorded regardless of which branch this validator resolves on,
        # not only the bare-shape-match branch that originally needed it.
        candidate.state.repetition_count = count_occurrences(text, document_text)

        has_dictionary_evidence = any(
            d.metadata.get("pattern", "").startswith("dictionary") for d in detections
        )
        has_title_evidence = any(d.metadata.get("pattern") == "titled" for d in detections)
        has_spacy_evidence = any(d.detector == DetectorName.SPACY for d in detections)

        if has_dictionary_evidence or has_title_evidence:
            return ValidationResult(
                self.name, passed=True, score=0.0, message="Has corroborating evidence beyond bare shape-match",
            )

        spacy_only_ambiguous = has_spacy_evidence and self._starts_with_ambiguous_word(text, knowledge_base)
        if has_spacy_evidence and not spacy_only_ambiguous:
            return ValidationResult(
                self.name, passed=True, score=0.0, message="Has corroborating evidence beyond bare shape-match",
            )

        repeat_count = candidate.state.repetition_count
        if repeat_count >= self.MIN_REPETITION_FOR_CORROBORATION:
            return ValidationResult(
                self.name, passed=True, score=0.0,
                message=f"Recurs {repeat_count}x as an identical bare match in this document - "
                        f"repetition itself treated as corroboration",
            )

        if spacy_only_ambiguous:
            first_word = text.split()[0]
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"spaCy is the only evidence and '{first_word}' is a known first-name/"
                        f"common-noun collision word - not trusted alone without dictionary, "
                        f"title, or {self.MIN_REPETITION_FOR_CORROBORATION}+ repetition",
            )

        return ValidationResult(
            self.name, passed=False, severity="hard",
            message="Only a bare regex shape-match with no dictionary, title, or spaCy "
                    "corroboration - too weak to be worth a human's review",
        )

    @staticmethod
    def _starts_with_ambiguous_word(text: str, knowledge_base: KnowledgeBase) -> bool:
        tokens = text.split()
        return bool(tokens) and knowledge_base.is_ambiguous_first_name(tokens[0])
