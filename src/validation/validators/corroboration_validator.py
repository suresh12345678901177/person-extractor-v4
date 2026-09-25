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

A fourth refinement, added after a real production false positive
(2026-09-16, real case scan): a single-token DICTIONARY hit
has exactly the same blind spot as spaCy above, for a related but
distinct reason. "Major" is a genuine (if rare) entry in
assets/first_names/first_names.txt, but it is also an ordinary English
rank/role word that appears constantly in non-narrative technical text
(Android dumpsys/log dumps, report boilerplate) with zero relation to a
person. Because a lone "Major" produces a dictionary_single_token
detection, it used to satisfy this validator immediately (any
"dictionary*" pattern short-circuited straight to pass) and then clear
DecisionEngine's knowledge-corroboration gate via the low
ML_WEAK_DICTIONARY_VETO_FLOOR - real measured impact: ACCEPTED 63 times
across 6 unrelated files, zero of them an actual person. The fix
generalizes the SAME ambiguous-word list used for spaCy above: a
single-token dictionary hit whose token is in
ambiguous_first_names.txt is no longer trusted alone either - it now
needs the same repetition-or-title-or-window-dictionary corroboration
any other ambiguous or bare match needs. A genuine person whose name
IS one of these words (a real "Major Grant") is still fully reachable
via in-document repetition, a title match ("Mr. Major"), or full
first+last dictionary window support - this only removes the single
bare-token dictionary hit's ability to corroborate ON ITS OWN.

A fifth refinement, added 2026-09-17 after real casework: a single
Android locale/calendar-picker resource-dump file ("Jan", "Sep",
"Domingo", "Julio", "Maj"...) got ACCEPTED 10+ times each purely because
these words are ALSO genuine dictionary first names in some locale
(real people are named "Jan" or "Domingo") and repeat heavily in the
file - exactly the signal the fourth refinement's repetition allowance
was built to trust. For an ordinary ambiguous word (like "Major"), heavy
repetition IS meaningful: a real recurring person is genuinely more
likely to be repeated than a distractor phrase (see the "Threshold
lowered 10 -> 4" measurement above). But a calendar word is structurally
different - a locale resource file's fixed ~19-entry vocabulary
(days/months) is EXPECTED to repeat many times regardless of whether any
person is involved, so repetition is actively misleading evidence for
this specific, narrow word class, not corroboration at all. Calendar
words (assets/common_words/multilingual_calendar_words.txt) therefore
skip the repetition-corroboration allowance entirely and require a
title or full first+last dictionary window match - the same bar as an
ambiguous word that has ALSO exhausted its repetition allowance. A
genuine person whose name is a calendar word (a real "Julio Mendez") is
still fully reachable via a title match or full dictionary window
support, exactly as before - only the repetition-as-corroboration
shortcut is removed for this word class specifically.

A sixth refinement, added 2026-09-17: the SAME structural gap generalizes
well beyond calendar words. StructureValidator hard-rejects ALL-CAPS
tokens as acronym-shaped UNLESS the token happens to match a known LAST
name - an escape hatch meant for genuine surnames written in caps (legal
documents, formal Indian ID-style formatting). But at this project's
dictionary scale (21,000+ last names across 81 locales), plenty of
ordinary acronyms/tech terms/legal-boilerplate words coincidentally ARE
someone's rare real surname somewhere - confirmed directly this session:
"READ", "HANDLER", "KNOX", "POL", "FOA", "UUS", "SHA", "WILD" (Android
dumpsys/log terms) and "LAW" (from repeated "GOVERNING LAW" EULA
section headers - real source: eula_12.txt, 5 occurrences in that one
file alone) all cleared StructureValidator this way, then ACCEPTED via
has_single_token_dictionary_evidence + the low ML_WEAK_DICTIONARY_VETO_
FLOOR. These words repeat because they are boilerplate/technical
vocabulary that inherently recurs (a EULA's "law" clause repeats across
many similar EULA files; a log identifier repeats every time that code
path runs) - the exact same "repetition reflects the content's
structure, not a real recurring person" pattern as calendar words, so
they get the identical treatment: an ALL-CAPS acronym-shaped SINGLE
token that only passed structure validation via the last-name coincidence
skips the repetition-corroboration allowance and needs a title or full
first+last dictionary window match instead. A genuine all-caps single
name (a real "LAW" used as someone's surname, standing alone) is still
reachable via a title ("Mr. LAW") - this only removes the repetition/
weak-single-token-dictionary shortcut for this specific, narrow,
structurally-risky class.
"""

from __future__ import annotations

import re

from src.candidate.name_propagation import DOCUMENT_NAME_PATTERN
from src.core.models import CandidateResult, DetectorName, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.latin import LATIN_UPPER
from src.preprocessing.segmenter import count_occurrences
from src.validation.base_validator import BaseValidator

_ACRONYM_SHAPE_RE = re.compile(rf"^[{LATIN_UPPER}]{{2,}}s?$")  # accented capitals too - see structure_validator.py


class CorroborationValidator(BaseValidator):
    name = "corroboration"

    MIN_REPETITION_FOR_CORROBORATION = 4

    def __init__(self) -> None:
        # Per-document memo cache for count_occurrences(text, document_text)
        # - a full-document regex scan, so O(document_length) per call.
        # Called once per surviving candidate, but real documents (and
        # especially structured non-prose dumps: one real 1.6MB case file
        # had 5937 candidates collapsing to just 5 distinct texts) have
        # massive duplication in candidate text, so memoizing by text
        # turns O(candidates x document_length) into O(distinct_texts x
        # document_length) - on that file, ~9 minutes -> 0.14s. Must be
        # cleared per document (see reset_document_cache) since this
        # validator instance is reused across every file in a batch run.
        self._repetition_cache: dict[str, int] = {}

    def reset_document_cache(self) -> None:
        self._repetition_cache.clear()

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        detections = candidate.candidate.source_detections
        text = candidate.candidate.normalized_text

        # Computed unconditionally, even on the branches below that don't
        # need it to decide pass/fail - DecisionEngine's ML-high-confidence
        # knowledge-corroboration path reuses this exact count later (see
        # CandidateState.repetition_count's docstring), so it must be
        # recorded regardless of which branch this validator resolves on,
        # not only the bare-shape-match branch that originally needed it.
        if text not in self._repetition_cache:
            self._repetition_cache[text] = count_occurrences(text, document_text)
        candidate.state.repetition_count = self._repetition_cache[text]

        has_window_dictionary_evidence = any(
            d.metadata.get("pattern") == "dictionary" for d in detections
        )
        has_single_token_dictionary_evidence = any(
            d.metadata.get("pattern") == "dictionary_single_token" for d in detections
        )
        has_title_evidence = any(d.metadata.get("pattern") == "titled" for d in detections)
        has_spacy_evidence = any(d.detector == DetectorName.SPACY for d in detections)
        # First name of a full name this document already accepted - see
        # src/candidate/name_propagation.py (which never propagates
        # ambiguous/common/calendar words, the classes this validator's
        # refinements below exist for).
        has_document_name_evidence = any(
            d.metadata.get("pattern") == DOCUMENT_NAME_PATTERN for d in detections
        )

        if has_window_dictionary_evidence or has_title_evidence or has_document_name_evidence:
            return ValidationResult(
                self.name, passed=True, score=0.0, message="Has corroborating evidence beyond bare shape-match",
            )

        # A single-token dictionary hit or spaCy tag is normally enough on
        # its own - UNLESS the token is a known first-name/common-word
        # collision (assets/ambiguous_words/ambiguous_first_names.txt),
        # in which case EITHER signal alone is exactly as unreliable as
        # the other (see module docstring's third and fourth refinements)
        # and both fall through to the same repetition-or-reject path.
        is_ambiguous_collision = self._starts_with_ambiguous_word(text, knowledge_base)
        is_calendar_word = len(text.split()) == 1 and knowledge_base.is_calendar_word(text)
        is_allcaps_last_name_collision = self._is_allcaps_last_name_collision(text, knowledge_base)
        has_non_ambiguous_corroboration = (
            has_single_token_dictionary_evidence or has_spacy_evidence
        ) and not is_ambiguous_collision and not is_calendar_word and not is_allcaps_last_name_collision
        if has_non_ambiguous_corroboration:
            return ValidationResult(
                self.name, passed=True, score=0.0, message="Has corroborating evidence beyond bare shape-match",
            )

        # Calendar words skip the repetition-corroboration allowance below
        # entirely - see module docstring's fifth refinement: unlike an
        # ordinary ambiguous word, a locale resource file's fixed
        # day/month vocabulary is EXPECTED to repeat heavily regardless of
        # personhood, so repetition is misleading evidence here, not
        # corroboration.
        if is_calendar_word:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"'{text}' is a recognized calendar word (day/month name) with only a "
                        f"single-token dictionary hit or spaCy tag as evidence - repetition does not "
                        f"count as corroboration for this word class (a locale resource file's fixed "
                        f"vocabulary is expected to repeat regardless of personhood); needs a title or "
                        f"full first+last dictionary window match instead",
            )

        # ALL-CAPS acronym-shaped single tokens that only passed
        # StructureValidator via a coincidental last-name match skip the
        # repetition-corroboration allowance too - see module docstring's
        # sixth refinement: these repeat because they're boilerplate/
        # technical vocabulary (EULA clauses, log identifiers), not
        # because a real person recurs.
        if is_allcaps_last_name_collision:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"'{text}' is ALL-CAPS and acronym-shaped, only structurally valid because it "
                        f"coincidentally matches a known last name - repetition does not count as "
                        f"corroboration for this word class (boilerplate/technical terms recur "
                        f"regardless of personhood); needs a title or full first+last dictionary "
                        f"window match instead",
            )

        repeat_count = candidate.state.repetition_count
        if repeat_count >= self.MIN_REPETITION_FOR_CORROBORATION:
            return ValidationResult(
                self.name, passed=True, score=0.0,
                message=f"Recurs {repeat_count}x as an identical bare match in this document - "
                        f"repetition itself treated as corroboration",
            )

        if is_ambiguous_collision and (has_single_token_dictionary_evidence or has_spacy_evidence):
            first_word = text.split()[0]
            evidence_desc = "spaCy" if has_spacy_evidence and not has_single_token_dictionary_evidence else \
                "a single-token dictionary hit" if has_single_token_dictionary_evidence and not has_spacy_evidence else \
                "spaCy and a single-token dictionary hit"
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"{evidence_desc} is the only evidence and '{first_word}' is a known first-name/"
                        f"common-noun collision word - not trusted alone without a title, a full "
                        f"first+last dictionary window match, or {self.MIN_REPETITION_FOR_CORROBORATION}+ "
                        f"repetition",
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

    @staticmethod
    def _is_allcaps_last_name_collision(text: str, knowledge_base: KnowledgeBase) -> bool:
        tokens = text.split()
        if len(tokens) != 1:
            return False
        token = tokens[0]
        letters_only = token.replace("-", "").rstrip(".")
        is_acronym_shaped = bool(_ACRONYM_SHAPE_RE.match(letters_only))
        return is_acronym_shaped and knowledge_base.is_known_last_name(token)
