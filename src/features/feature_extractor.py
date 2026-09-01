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
from src.preprocessing.segmenter import following_char, is_inside_quotes, preceding_word

_INITIAL_RE = re.compile(r"^[A-Z]\.?$")


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
    )
