"""
src.preprocessing.language_filter
====================================
A cheap, fully offline gate that flags documents unlikely to be
natural-language English prose - non-English text, translated EULAs,
phone-settings screens, and other boilerplate that sometimes rides
alongside real evidence in a forensic case export - so the pipeline
can skip running its (English-only) detectors on them instead of
producing garbage person-name candidates.

Root cause this exists for: assets/first_names.txt and
assets/last_names.txt are global, multi-culture name lists, so short
real surnames like "Ali", "Kak", "Kar", "Pal", "Tak" inevitably
collide with ordinary function words in other languages (e.g.
Slovenian "ali" = "or"). On a page of foreign-language legal
boilerplate those collisions repeat dozens of times, and the
repetition-based corroboration signal that correctly confirms a real
name in an English chat thread instead reinforces the false positive.
No amount of dictionary curation removes every such cross-language
collision - filtering by language before detection ever runs is the
actual fix.

Reuses the existing assets/stopwords/stopwords.txt list (969 common
English words, already loaded by KnowledgeBase for boundary trimming)
rather than adding a new wordlist or a language-ID dependency: the
fraction of a document's words that land in that list cleanly
separates English prose from other languages. Measured directly
against real files: 0.30-0.49 on every file in datasets/benchmark/
(English chat/email prose) vs. 0.01-0.05 on five non-English documents
pulled from a real ProDiscover case export (Slovenian/Albanian/Swedish
Samsung EULAs) - the threshold below is set well clear of both
clusters, not guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.knowledge.knowledge_base import KnowledgeBase

_WORD_RE = re.compile(r"[A-Za-z]+")

# Below this many words there isn't enough signal to judge language
# reliably. Always treated as English below this size, since a false
# skip (silently dropping a short genuine English message) costs more
# than occasionally running detection on a short non-English one - which
# rarely has enough candidates to matter anyway.
MIN_WORDS_FOR_LANGUAGE_CHECK = 30

# See module docstring for how this was calibrated.
ENGLISH_WORD_RATIO_THRESHOLD = 0.15


def is_short_non_prose(text: str, knowledge_base: KnowledgeBase) -> bool:
    """True for a document under MIN_WORDS_FOR_LANGUAGE_CHECK Latin-letter
    words whose stopword share is also below ENGLISH_WORD_RATIO_THRESHOLD:
    too short for check_english() to skip, and not reading as an English
    sentence either - an OCR scrap of an image, a lone word, a terse data
    record. A short genuine note ("Call Rajesh tomorrow about the payment",
    ratio 0.67) is not. See person_extractor._cap_lone_single_tokens."""
    words = _WORD_RE.findall(text)
    if len(words) >= MIN_WORDS_FOR_LANGUAGE_CHECK:
        return False
    return not words or sum(1 for w in words if knowledge_base.is_stopword(w)) / len(words) < ENGLISH_WORD_RATIO_THRESHOLD


@dataclass(frozen=True, slots=True)
class LanguageCheck:
    is_english: bool
    english_word_ratio: float
    word_count: int


def check_english(text: str, knowledge_base: KnowledgeBase) -> LanguageCheck:
    words = _WORD_RE.findall(text)
    if len(words) < MIN_WORDS_FOR_LANGUAGE_CHECK:
        return LanguageCheck(is_english=True, english_word_ratio=1.0, word_count=len(words))

    hits = sum(1 for w in words if knowledge_base.is_stopword(w))
    ratio = hits / len(words)
    return LanguageCheck(
        is_english=ratio >= ENGLISH_WORD_RATIO_THRESHOLD,
        english_word_ratio=round(ratio, 4),
        word_count=len(words),
    )
