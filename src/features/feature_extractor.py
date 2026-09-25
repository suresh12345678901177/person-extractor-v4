"""
src.features.feature_extractor
=================================
Computes a FeatureVector for a CandidateResult. This is the ONE place
that turns a candidate + its document context into the numeric/
categorical features the LightGBM classifier consumes - kept separate
from both detection (which produces raw evidence) and validation (which
produces pass/fail + soft scores), per the blueprint's Feature
Engineering layer and single-responsibility principle.
"""

from __future__ import annotations

import re

from src.core.models import CandidateResult, DetectorName, FeatureVector
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.latin import LATIN_UPPER
from src.preprocessing.segmenter import (
    following_char,
    following_words,
    is_inside_quotes,
    preceding_word,
    preceding_words,
)

_INITIAL_RE = re.compile(rf"^[{LATIN_UPPER}]\.?$")

# Local-context cue words for the person/non-person context-window
# features, added 2026-09-17 after an accuracy review found the model
# never received any signal about the candidate's surrounding words -
# only shape/dictionary/confidence numbers, which cannot distinguish
# "Marcus Webb" from "Convertible Debt" when those properties happen to
# match. Deliberately small, hand-curated lists (not the full stopword/
# dictionary assets) - kept narrow and high-precision so these features
# stay genuinely informative rather than firing on nearly everything.
_PERSON_CONTEXT_CUES = frozenset({
    "said", "says", "saying", "told", "tells", "telling",
    "met", "meets", "meeting", "contacted", "contact",
    "interviewed", "according", "regards", "sincerely",
    "spoke", "speaking", "signed", "thanked", "greeted",
    "asked", "replied", "confirmed", "reported",
})

# Reuses OrganizationValidator's own keyword list (single source of
# truth for "this word means organization/institution, not person")
# plus a small supplement of the specific business/product/UI terms
# actually found in real false positives this session (Convertible
# Debt, Windows Division, Windows Embedded, Display Overlays, EventHub
# Devices, Start LockScreen, Submit Complaint - see decision_engine.py's
# design principle 3 and generate_training_data.py's
# CHAT_HEADER_NEGATIVE_PHRASES).
from src.validation.validators.organization_validator import ORG_KEYWORDS as _ORG_KEYWORDS

_NONPERSON_CONTEXT_CUES = _ORG_KEYWORDS | frozenset({
    "debt", "embedded", "overlays", "devices", "lockscreen", "complaint",
    "settings", "version", "policy", "invoice", "agreement", "contract",
    "account", "capital", "holdings", "insurance",
})


def text_only_features(text: str, knowledge_base: KnowledgeBase) -> tuple[float, int, int]:
    """(common_word_ratio, first_token_is_ambiguous, place_or_org_token_count)
    for a candidate's normalized text - see FeatureVector for why these
    exist. Depends on the text alone (no document context), which is
    what lets scripts/migrate_training_data_add_text_features.py back-fill
    them exactly for stored training rows; that script calls THIS
    function, so stored and live values can never drift apart.

    Title/honorific tokens ("Dr.", "Mr.") are excluded first - they're
    already their own features, and "Mr" would otherwise count as an
    ordinary word and dilute the ratio for exactly the titled names that
    are most certainly people."""
    tokens = [t.rstrip(".'’") for t in text.split()]
    body = [t for t in tokens if t and not knowledge_base.is_title(t) and not knowledge_base.is_honorific(t)]
    if not body:
        body = [t for t in tokens if t]
    if not body:
        return 0.0, 0, 0
    common_ratio = sum(1 for t in body if knowledge_base.is_common_word(t)) / len(body)
    first_ambiguous = int(knowledge_base.is_ambiguous_first_name(body[0]))
    place_or_org = sum(
        1 for t in body if knowledge_base.is_location(t) or knowledge_base.is_organization(t)
    )
    return round(common_ratio, 4), first_ambiguous, place_or_org


def extract_features(
    candidate: CandidateResult, document_text: str, knowledge_base: KnowledgeBase
) -> FeatureVector:
    text = candidate.candidate.normalized_text
    tokens = text.split()
    start, end = candidate.candidate.start, candidate.candidate.end

    first_hits = sum(1 for t in tokens if knowledge_base.is_known_first_name(t))
    last_hits = sum(1 for t in tokens if knowledge_base.is_known_last_name(t))
    dict_hit_ratio = (first_hits + last_hits) / len(tokens) if tokens else 0.0
    capitalized_ratio = sum(1 for t in tokens if t[:1].isupper()) / len(tokens) if tokens else 0.0
    initial_count = sum(1 for t in tokens if _INITIAL_RE.match(t))

    detections = candidate.candidate.source_detections
    # has_title was previously ONLY true when the regex detector's own
    # hardcoded title list (see regex_detector.py's _TITLES) produced the
    # "titled" pattern - meaning assets/titles/titles.txt was loaded by
    # KnowledgeBase but never actually consulted anywhere. This also
    # checks the dictionary-driven list, same pattern as has_honorific,
    # so expanding titles.txt has real effect on the ML feature instead
    # of being inert data.
    has_title = any(d.metadata.get("pattern") == "titled" for d in detections) or any(
        knowledge_base.is_title(t.rstrip(".")) for t in tokens
    )
    has_honorific = any(
        knowledge_base.is_honorific(t.rstrip(".")) for t in tokens
    )

    regex_conf = max((d.confidence for d in detections if d.detector == DetectorName.REGEX), default=0.0)
    dict_conf = max((d.confidence for d in detections if d.detector == DetectorName.DICTIONARY), default=0.0)
    spacy_conf = max((d.confidence for d in detections if d.detector == DetectorName.SPACY), default=0.0)
    detector_count = len({d.detector for d in detections})

    prev_word = preceding_word(document_text, start)
    prev_is_stopword = bool(prev_word) and knowledge_base.is_stopword(prev_word)

    next_char = following_char(document_text, end)
    next_is_punct = bool(next_char) and not next_char.isalnum()

    in_quotes = is_inside_quotes(document_text, start, end)

    prev_words = [w.lower() for w in preceding_words(document_text, start, n=3)]
    preceding_is_person_cue = any(w in _PERSON_CONTEXT_CUES for w in prev_words)

    next_words = [w.lower() for w in following_words(document_text, end, n=3)]
    following_is_nonperson_cue = any(w.rstrip(".,") in _NONPERSON_CONTEXT_CUES for w in next_words)

    # "Alone on its own line" (the real chat/email sender-header shape,
    # see following_char()'s docstring): nothing else meaningful before
    # or after the candidate on the same line.
    is_isolated = (not prev_words) and (not next_words)

    common_ratio, first_ambiguous, place_or_org = text_only_features(text, knowledge_base)

    return FeatureVector(
        token_count=len(tokens),
        char_length=len(text),
        has_title=int(has_title),
        has_honorific=int(has_honorific),
        first_name_dict_hits=first_hits,
        last_name_dict_hits=last_hits,
        dict_hit_ratio=round(dict_hit_ratio, 4),
        capitalized_ratio=round(capitalized_ratio, 4),
        initial_token_count=initial_count,
        regex_confidence=round(regex_conf, 4),
        dictionary_confidence=round(dict_conf, 4),
        spacy_confidence=round(spacy_conf, 4),
        detector_count=detector_count,
        preceding_word_is_stopword=int(prev_is_stopword),
        following_char_is_punct=int(next_is_punct),
        occurs_in_quotes=int(in_quotes),
        preceding_context_is_person_cue=int(preceding_is_person_cue),
        following_context_is_nonperson_cue=int(following_is_nonperson_cue),
        is_isolated_line=int(is_isolated),
        common_word_ratio=common_ratio,
        first_token_is_ambiguous=first_ambiguous,
        place_or_org_token_count=place_or_org,
    )
