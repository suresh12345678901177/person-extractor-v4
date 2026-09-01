"""Hard-rejects candidates matching a known slogan/banner/campaign
phrase, checked bidirectionally as sub-phrases, PLUS a generic
marker-word check that catches NOVEL slogans never seen before
("Bharat Mata Ki Jai", "Plastic Free India") without needing an exact
match. Exact-phrase matching alone is a losing battle against novel
slogans; the marker-word list generalizes, at the acceptable cost of
occasionally rejecting a legitimate name containing one of these words
(rare in practice, and precision is the priority for this project)."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

SLOGAN_MARKER_WORDS = frozenset({
    "india", "bharat", "jai", "vande", "mataram", "abhiyan", "swachh",
    "tiranga", "ghar", "kisan", "mata", "hind", "ki",
    "digital", "skill", "mission", "clean", "free", "vote", "save",
    "plastic", "responsibly", "sale", "opening", "admission", "open",
    "green", "safety", "awareness", "welcome",
    # Bulk addition: common Indian government-scheme vocabulary (source:
    # Wikipedia "List of government schemes in India", CC-BY-SA 4.0),
    # generalizes to scheme names never explicitly listed in
    # campaigns.txt. Cross-checked against first_names.txt/last_names.txt
    # and excluded any word that collides with a real personal name -
    # "vikas", "pradhan", "nidhi", "kaushal", "jyoti" are all common
    # Indian given/surnames as well as scheme-name words, so they are
    # deliberately left OUT of this hard-reject list even though they'd
    # improve slogan recall, per this validator's own stated
    # precision-first rationale.
    "yojana", "yojna", "karyakram", "gram", "suraksha", "bima", "awas",
    "shiksha", "swasthya", "rozgar", "mudra", "ujjwala", "sinchai",
    "fasal", "matritva", "annadata",
})


class CampaignValidator(BaseValidator):
    name = "campaign"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        text = candidate.candidate.normalized_text.lower()

        if text in knowledge_base.campaigns:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"'{candidate.candidate.normalized_text}' matches known slogan/campaign phrase",
            )

        # Word-boundary-safe sub-phrase check (not raw character-substring
        # containment - see LocationValidator for why that silently breaks
        # once a phrase list grows: e.g. "in" would wrongly match inside
        # "India" as a substring).
        raw_tokens = candidate.candidate.normalized_text.split()
        for start in range(len(raw_tokens)):
            for end in range(start + 1, len(raw_tokens) + 1):
                sub_phrase = " ".join(raw_tokens[start:end]).lower()
                if sub_phrase == text:
                    continue
                if sub_phrase in knowledge_base.campaigns:
                    return ValidationResult(
                        self.name, passed=False, severity="hard",
                        message=f"Contains known slogan/campaign phrase: '{sub_phrase}'",
                    )

        tokens = candidate.candidate.normalized_text.split()
        marker_tokens = [t for t in tokens if t.lower().rstrip(".,") in SLOGAN_MARKER_WORDS]
        if marker_tokens:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"Contains slogan/campaign marker word(s): {', '.join(marker_tokens)}",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="Not a known slogan/campaign")
