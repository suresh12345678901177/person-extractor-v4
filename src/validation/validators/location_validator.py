"""
src.validation.validators.location_validator
================================================
Hard-rejects candidates matching a known place name (city, region,
country - e.g. "Hyderabad", "Telangana", "New Delhi"). Kept separate
from BlacklistValidator (which holds generic non-person strings) so the
location list can be maintained/extended independently by anyone
editing `assets/locations/locations.txt` without touching unrelated
blacklist entries - single responsibility per the blueprint's Design
Principles.

Checked as word-boundary-safe sub-phrases (same approach as
BlacklistValidator): a location can appear with extra words attached by
the regex detector ("the New Delhi office"), so every contiguous
sub-phrase of the candidate's own tokens is checked against the known
locations set, not just the full candidate text.

Deliberately NOT a raw character-substring check ("phrase in text or
text in phrase"), even though that was this validator's original
implementation - at a hand-picked list of ~30 locations that happened to
never collide, but it silently breaks once the location list grows past
a few thousand entries. A real regression found during testing: adding
world-cities data made "Anil" get hard-rejected as a location, because
the substring "anil" happens to occur inside "Manila" and "Manzanillo".
Character containment is not the same thing as word overlap.

A second collision class found on real casework (case 930204590): a
person's actual SURNAME can independently also be a real place name
("Bello" - a city in Colombia and a district in Nigeria - is also a
common West African surname). A blanket hard-reject on any location
sub-phrase match silently erased every mention of "Fatima Bello" across
an entire case, with no way for an analyst to even see it happened.
_is_rescued() narrowly rescues exactly this shape: some other token in
the candidate must independently be a known first name (the
location-matching token itself no longer has to ALSO be a listed last
name - see that method's docstring for why that earlier requirement
defeated the rescue's own purpose for out-of-dictionary surnames). This
deliberately does not cover a bare first-name/place homonym standing
alone ("Jordan", "Chad", "Sydney" as a lone token) - those stay
hard-rejected, since without an accompanying known-first-name token
there's nothing distinguishing the mention from an actual place
reference.

A third collision class found on real casework (case 930204601): the
FIRST name, not the surname, can be the one that collides with a place
("Charlotte" - a major US city - as a first name paired with a
completely out-of-dictionary surname "Reyes"). The dictionary-based
rescue above cannot help here at all when NEITHER token is in any
dictionary - there is no known first/last name anywhere in the
candidate to corroborate it. Silently erased all 61 mentions of
"Charlotte Reyes" in that case, the single most-mentioned person in the
document, with zero trace anywhere (not even REVIEW). _is_rescued() also
accepts document-wide repetition as a fallback here - same "does this
exact form recur far more than chance in THIS document" principle
CorroborationValidator already uses to gate the same "no dictionary
support at all" case elsewhere in this pipeline (see that validator's
module docstring). Deliberately reuses the SAME repetition bar
(MIN_REPETITION_FOR_CORROBORATION), not a separately-invented number, so
"how much repetition counts as trust" stays answered in exactly one
place. A one-off or rarely-repeated location/name collision with no
dictionary support anywhere still gets no rescue and stays
hard-rejected, consistent with every other no-evidence-at-all case in
this pipeline.
"""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.segmenter import count_occurrences
from src.validation.base_validator import BaseValidator
from src.validation.validators.corroboration_validator import CorroborationValidator


class LocationValidator(BaseValidator):
    name = "location"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        text = candidate.candidate.normalized_text
        tokens = text.split()

        if knowledge_base.is_location(text) and not self._is_rescued(tokens, 0, len(tokens), knowledge_base, candidate, document_text):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"'{text}' matches known location",
            )

        for start in range(len(tokens)):
            for end in range(start + 1, len(tokens) + 1):
                sub_phrase = " ".join(tokens[start:end])
                if sub_phrase == text:
                    continue
                if knowledge_base.is_location(sub_phrase) and not self._is_rescued(tokens, start, end, knowledge_base, candidate, document_text):
                    return ValidationResult(
                        self.name, passed=False, severity="hard",
                        message=f"Contains known location: '{sub_phrase}'",
                    )

        return ValidationResult(self.name, passed=True, score=0.0, message="Not a known location")

    @staticmethod
    def _is_rescued(
        tokens: list[str], start: int, end: int, knowledge_base: KnowledgeBase,
        candidate: CandidateResult, document_text: str,
    ) -> bool:
        """A location sub-phrase is rescued from hard-rejection when
        another token in the SAME candidate is a known first name - the
        location-matching token then reads as functioning as a surname
        in this specific candidate, not as an actual place reference.
        Previously also required the location-matching text itself to
        independently be a KNOWN last name, which meant this rescue only
        ever fired for surnames already in last_names.txt - defeating
        its own purpose for exactly the out-of-dictionary-surname case
        it exists for (found via testing: "Chidi Santos" - "Santos" is a
        known city AND a real surname, but wasn't in last_names.txt, so
        the rescue never fired and the whole name was hard-rejected).

        Falls back to document-wide repetition (same bar
        CorroborationValidator uses) when NEITHER token has any
        dictionary support at all - see this module's docstring for the
        "Charlotte Reyes" case this covers, where the dictionary-based
        rescue above structurally cannot apply.

        Still does not cover a bare first-name/place homonym standing
        alone ("Jordan", "Chad", "Sydney" as a lone token) - those stay
        hard-rejected, since without an accompanying token there's
        nothing distinguishing the mention from an actual place
        reference.
        """
        other_tokens = tokens[:start] + tokens[end:]
        if any(knowledge_base.is_known_first_name(t.rstrip(".")) for t in other_tokens):
            return True
        if not other_tokens:
            return False

        # Repetition of the FULL candidate ("Charlotte Reyes"), not just
        # the matched sub-phrase ("Charlotte") - a lone repeated city
        # name is still a location, not a rescue signal. Normally already
        # computed by CorroborationValidator (runs earlier in the hard-
        # validator order); recomputed directly only as a fallback for
        # --loose-gate mode, where that validator is skipped entirely.
        full_text = candidate.candidate.normalized_text
        repeat_count = candidate.state.repetition_count or count_occurrences(full_text, document_text)
        return repeat_count >= CorroborationValidator.MIN_REPETITION_FOR_CORROBORATION
