"""Validates token shape: starts with a capital letter, letters/hyphens/
apostrophes only, and is not an ALL-CAPS acronym or acronym-plural/
hyphenated-acronym-fragment (e.g. 'CCTV', 'IDs', 'TXN-AE')."""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.latin import LATIN_LETTERS, LATIN_UPPER
from src.validation.base_validator import BaseValidator

# Latin letters incl. accents (src/preprocessing/latin.py) - the old
# ASCII-only shape hard-rejected every accented name ("José", "Björn").
_VALID_TOKEN_RE = re.compile(rf"^[{LATIN_UPPER}][{LATIN_LETTERS}'\-]*\.?$")
# Latin capitals incl. accented ones (2026-09-24): ASCII-only [A-Z] let
# accented all-caps tokens skip the acronym check entirely - e.g. "ÑËÙ",
# decoded out of an Android binary XML file, passed as a valid name token,
# while "JOHN" (unknown as a last name) would have been rejected.
_ACRONYM_RE = re.compile(rf"^[{LATIN_UPPER}]{{2,}}$")
_ACRONYM_PLURAL_RE = re.compile(rf"^[{LATIN_UPPER}]{{2,}}s$")


class StructureValidator(BaseValidator):
    name = "structure"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()

        if not tokens:
            return ValidationResult(self.name, passed=False, severity="hard", message="Empty candidate")

        for token in tokens:
            if not _VALID_TOKEN_RE.match(token):
                return ValidationResult(
                    self.name, passed=False, severity="hard",
                    message=f"Token '{token}' does not match name-token shape",
                )

            letters_only = token.replace("-", "").rstrip(".")
            is_acronym = _ACRONYM_RE.match(letters_only) or _ACRONYM_PLURAL_RE.match(letters_only)
            # EXACT last-name lookup, deliberately not is_known_last_name():
            # its accent-variant fallback folds "ÑËÙ" to the real surname
            # "neu", which would let accent-stuffed all-caps garbage through
            # this all-caps-surname exemption.
            if is_acronym and token.lower() not in knowledge_base.last_names:
                return ValidationResult(
                    self.name, passed=False, severity="hard",
                    message=f"Token '{token}' looks like an acronym, not a name",
                )

            irregular = _irregular_capitals(token, knowledge_base)
            if irregular:
                return ValidationResult(
                    self.name, passed=False, severity="hard",
                    message=f"Token '{token}' {irregular} - machine identifier/garbage, not a name",
                )

        start, end = candidate.candidate.start, candidate.candidate.end
        if _MOJIBAKE_RE.search(document_text[max(0, start - 1):end + 1]):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Text around the candidate is UTF-8 read as Latin-1 (e.g. 'Ã¡') - "
                        "encoding garbage, not a name",
            )

        return ValidationResult(self.name, passed=True, score=0.10, message="Structure OK")


# Added 2026-09-24 after a real case scan (Android extraction) accepted
# machine-text fragments as names once accented letters became valid name
# characters: mixed-case identifiers ("ToMs", "KeR", "LUt", "DeX"),
# CamelCase compounds ("DisplayType", "KeyCharacterMapFile") and mojibake
# ("TomÃ" from "TomÃ¡s" = "Tomás" misdecoded).
_NAME_PREFIXES = frozenset({
    "mc", "mac", "de", "del", "della", "der", "des", "di", "da", "dal", "du", "la", "le", "lo",
    "van", "von", "st", "fitz", "al", "el", "ben", "bin", "abu",
})
_SEGMENT_RE = re.compile(rf"[{LATIN_UPPER}][^{LATIN_UPPER}]*")
# A UTF-8 lead byte (U+00C2-U+00DF when misread as Latin-1) directly
# followed by a continuation byte (U+0080-U+00BF).
_MOJIBAKE_RE = re.compile("[Â-ß][\u0080-¿]")


def _irregular_capitals(token: str, knowledge_base: KnowledgeBase) -> str | None:
    """Why a token's internal capitals make it a non-name, or None. Real
    names with internal capitals follow a prefix pattern - McDonald,
    MacArthur, DeShawn, LaToya, DelBene, JoAnn - so a segment is never a
    single letter, the last segment is 3+ letters unless the first is a
    known name prefix, and the segments aren't all ordinary English words."""
    for part in re.split(r"[-'’]", token.rstrip(".")):
        if len(part) < 2 or part.isupper() or not part[:1].isupper():
            continue
        segments = _SEGMENT_RE.findall(part)
        if len(segments) < 2:
            continue
        if any(len(s) < 2 for s in segments):
            return "mixes capitals irregularly"
        prefixed = segments[0].lower() in _NAME_PREFIXES
        if not prefixed and len(segments[-1]) < 3:
            return "mixes capitals irregularly"
        if not prefixed and all(knowledge_base.is_common_word(s) for s in segments):
            return "is ordinary words joined in CamelCase"
    return None
