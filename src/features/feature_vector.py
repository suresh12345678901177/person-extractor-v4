"""
src.features.feature_vector
==============================
Single source of truth for feature ORDER. Both the training script and
the live inference path import FEATURE_NAMES from here, so they can
never silently drift out of sync with each other (a classic, hard-to-
debug ML bug this module design prevents structurally rather than by
convention).
"""

from __future__ import annotations

# MUST match the field order of src.core.models.FeatureVector.as_list()
# exactly. Position N in this list names position N in every feature
# array the classifier ever sees, in training or inference.
FEATURE_NAMES: list[str] = [
    "token_count",
    "char_length",
    "has_title",
    "has_honorific",
    "first_name_dict_hits",
    "last_name_dict_hits",
    "dict_hit_ratio",
    "capitalized_ratio",
    "initial_token_count",
    "regex_confidence",
    "dictionary_confidence",
    "spacy_confidence",
    "detector_count",
    "preceding_word_is_stopword",
    "following_char_is_punct",
    "occurs_in_quotes",
    "preceding_context_is_person_cue",
    "following_context_is_nonperson_cue",
    "is_isolated_line",
]
