"""
src.preprocessing.latin
=========================
Single source of truth for "which letters can appear in a Latin-script
name", shared by every detector/validator that checks name-token shape.

Why this exists (found 2026-09-24): RegexDetector had been widened to
accept accented names, but StructureValidator, DictionaryDetector and
the initial checks still used ASCII-only [A-Z][a-zA-Z]* patterns - so
every accented name ("José García", "François Dubois", "Björn
Lindqvist") was detected and then hard-rejected as a malformed token,
never reaching even REVIEW. DictionaryDetector was worse than silent: it
matched the ASCII PREFIX of an accented word, so "François" produced a
confident dictionary hit on the fragment "Fran" (a real first name).
About 9,100 entries of the name lists (3,927 first names, 5,203 last
names) are accented and were unreachable.

Character classes are generated from Unicode data rather than typed as
ranges: the Latin Extended blocks interleave upper- and lowercase
(Ā ā Ă ă ...), so a range like "Ā-ſ" can't express "uppercase only" -
which is why the previous hand-written first-letter class [A-ZÀ-ÖØ-Þ]
silently excluded Ł, Š, Č, Ž (Polish/Czech/Croatian), Romanian Ș/Ț and
Vietnamese letters.

Covered: Basic Latin, Latin-1 Supplement, Latin Extended-A and -B, Latin
Extended Additional. Non-Latin scripts stay excluded on purpose - see
LanguageValidator's non-Latin-script hard reject.
"""

from __future__ import annotations

import unicodedata

_NON_ASCII_LATIN_BLOCKS = (
    (0x00C0, 0x00FF),  # Latin-1 Supplement letters (× and ÷ dropped by isalpha)
    (0x0100, 0x024F),  # Latin Extended-A and -B
    (0x1E00, 0x1EFF),  # Latin Extended Additional (Vietnamese, etc.)
)

_NON_ASCII_LATIN = "".join(
    chr(cp)
    for lo, hi in _NON_ASCII_LATIN_BLOCKS
    for cp in range(lo, hi + 1)
    if chr(cp).isalpha()
)

#: For use INSIDE a regex character class: any Latin letter.
LATIN_LETTERS = "A-Za-z" + _NON_ASCII_LATIN
#: For use INSIDE a regex character class: any uppercase Latin letter.
LATIN_UPPER = "A-Z" + "".join(ch for ch in _NON_ASCII_LATIN if ch.isupper())

# Letters that carry no combining mark to strip under NFKD but still
# have a conventional plain-ASCII spelling in names.
_FOLD_EXTRA = str.maketrans({
    "ß": "ss", "Æ": "Ae", "æ": "ae", "Œ": "Oe", "œ": "oe", "Ø": "O", "ø": "o",
    "Ł": "L", "ł": "l", "Đ": "D", "đ": "d", "Ð": "D", "ð": "d", "Þ": "Th",
    "þ": "th", "ı": "i", "Ħ": "H", "ħ": "h",
})


def fold_accents(text: str) -> str:
    """'José Müller-Łukasz' -> 'Jose Muller-Lukasz'. Used only as a
    FALLBACK dictionary lookup for tokens that are not plain ASCII (see
    KnowledgeBase), so ASCII text behaves exactly as before."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_FOLD_EXTRA))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
