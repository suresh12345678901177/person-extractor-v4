"""
scripts/generate_training_data.py
====================================
Generates a synthetic, distantly-supervised training set for the
LightGBM classifier by inserting known-positive spans (real names from
the knowledge base) and known-negative spans (organizations, locations,
campaign slogans, blacklist entries, acronyms, and generic capitalized
bigrams) into varied sentence templates, then running the SAME
`extract_features()` function used at inference time on each synthetic
candidate - guaranteeing zero train/inference feature skew.

This is a deliberate, documented distant-supervision technique (see
LightGBMClassifier's module docstring for the honesty note on what this
model actually learned). It is NOT a hand-labeled gold corpus.

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --scale 5
Produces:
    datasets/training/synthetic_training_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.models import Candidate, CandidateResult, CandidateState, Detection, DetectorName
from src.features.feature_extractor import _NONPERSON_CONTEXT_CUES, extract_features
from src.knowledge.knowledge_base import KnowledgeBase

random.seed(42)

TITLES_FOR_TEMPLATES = ["Dr.", "Mr.", "Mrs.", "Prof."]

POSITIVE_TEMPLATES = [
    "{name} attended the meeting on Tuesday.",
    "{name} presented the quarterly results.",
    "According to {name}, the project is on schedule.",
    "The report was signed by {name} last week.",
    "{name} confirmed the details over the phone.",
    "{name} and the rest of the team reviewed the contract.",
    "{name} said the timeline needed to change.",
    "A message from {name} arrived this morning.",
    "The committee thanked {name} for the update.",
    "{name} walked into the room and sat down.",
]

# Single-token positive templates: a lone first name is realistic (the
# same pattern our real dictionary_single_token detections cover), and
# without these the model never sees a token_count==1 positive example.
POSITIVE_SINGLE_TOKEN_TEMPLATES = [
    "{name} claimed he heard something outside.",
    "{name} insisted the meeting had already ended.",
    "According to {name}, the delivery was late.",
    "{name} was the last person to leave.",
]

# Chat/messaging sender-header templates: {name} sits ALONE on its own
# line, immediately followed by a newline then message text - the exact
# shape real WhatsApp/chat export files use for sender names (see
# segmenter.py's following_char() docstring: "a name sitting alone on
# its own chat-export line ... is exactly the real-world sender-header
# shape this tool's actual documents are full of"). Added 2026-09-16:
# until now the model only ever learned this shape from scarce REAL
# feedback examples, never from synthetic data, despite it being one of
# the most common patterns in this project's actual casework (chat-N.txt
# files). Deliberately LOWER spaCy-support probability than the other
# positive templates - matches the documented real finding that spaCy
# rarely tags a name on a bare, context-free standalone line (measured:
# ~9 of ~120 real appearances in one real chat-heavy document).
POSITIVE_CHAT_HEADER_TEMPLATES = [
    "{name}\nHey, are you free to talk later today?",
    "{name}\nI'll send the documents over by tomorrow morning.",
    "{name}\nCan we reschedule the call to 3pm?",
    "{name}\nThanks for confirming, see you then.",
    "{name}\nJust checking in on the status of this.",
    "{name}\nSure, that works for me.",
]

NEGATIVE_TEMPLATES = [
    "The banner outside read \"{name}\" in bright letters.",
    "A sign for {name} stood near the entrance.",
    "The announcement mentioned {name} repeatedly.",
    "Someone pointed at the {name} building across the street.",
    "The loudspeaker played \"{name}\" on a loop.",
    "Investigators found a reference to {name} in the notes.",
    "The poster displayed the words {name} in bold.",
    "A vehicle bearing the label {name} was parked outside.",
    "The receipt included the code {name} at the top.",
    "{name} appeared on the whiteboard without context.",
]

# Hard negatives: a REAL first name from the knowledge base combined
# with an organizational suffix word. Without these, "dict_hit_ratio"
# alone perfectly separates the synthetic classes and the model learns
# nothing beyond a dictionary lookup it already has - these force it to
# also weigh title/context/detector-agreement features.
ORG_SUFFIXES = [
    "Traders", "Enterprises", "Motors", "Textiles", "Constructions",
    "Industries", "Associates", "Exports", "Agencies", "Builders",
]
HARD_NEGATIVE_TEMPLATES = [
    "The company {name} opened a new branch downtown.",
    "A delivery van marked {name} was parked outside.",
    "Records show {name} was registered as a local business.",
    "The invoice was issued by {name} last month.",
]

GENERIC_CAPITALIZED_BIGRAMS = [
    "Great Wall", "Blue Ocean", "Fresh Market", "Open House", "Fast Track",
    "Main Street", "Grand Central", "North Star", "Silver Lake", "Golden Gate",
    "Early Bird", "High Tide", "Full Moon", "Night Shift", "Real Estate",
]

# Chat-header NEGATIVE counterpart to POSITIVE_CHAT_HEADER_TEMPLATES:
# a non-person phrase sitting alone on its own line, immediately
# followed by a newline - without this, every isolated-line example the
# model ever sees is a positive, risking it learning "alone on a line"
# itself as a person-signal rather than weighing it alongside the other
# features. Includes the exact real business/UI phrases found wrongly
# ACCEPTED in real casework (2026-09-16) - see
# src/decision/decision_engine.py's design principle 3.
NEGATIVE_CHAT_HEADER_TEMPLATES = [
    "{name}\nPlease hold while we connect you to an agent.",
    "{name}\nSettings updated successfully.",
    "{name}\nYour request has been submitted for review.",
    "{name}\nThis session will time out in 5 minutes.",
]
CHAT_HEADER_NEGATIVE_PHRASES = [
    "Vice President", "Convertible Debt", "Customer Service",
    "Windows Division", "Windows Embedded", "Display Overlays",
    "EventHub Devices", "Start LockScreen", "Submit Complaint",
]

# Context-cue reinforcement templates, added 2026-09-17 alongside the
# preceding_context_is_person_cue / following_context_is_nonperson_cue
# ML features (see feature_extractor.py's cue word lists). The plain
# POSITIVE_TEMPLATES/NEGATIVE_TEMPLATES above only incidentally vary
# these two features (2 of 10 positive templates happen to use a cue
# word, none of the negatives do) - not enough density for the model to
# learn a real weight for them distinct from what dict_hit_ratio already
# provides on its own. These examples deliberately pair a REAL
# dictionary name with a cue word, so the model must weigh the cue on
# top of (not instead of) existing dictionary evidence - exactly the
# "Marcus Webb" (real name, real person) vs "Marcus Webb Ltd" (the same
# real name, now clearly a business entity) distinction dict_hit_ratio
# alone cannot make.
CONTEXT_CUE_POSITIVE_TEMPLATES = [
    "According to {name}, the shipment already left.",
    "Investigators later interviewed {name} about the transfer.",
    "{name} confirmed the wire transfer this morning.",
    "The email was signed by {name} at the bottom.",
]
CONTEXT_CUE_NEGATIVE_TEMPLATES = [
    "The invoice was issued by {name} last month.",
    "Records show {name} was registered under that name.",
    "A payment was made to {name} on the 4th.",
    "The account documents list {name} at the top.",
]

# Unseen-name reinforcement, added 2026-09-17 after measuring directly:
# for 246 real, genuinely non-dictionary gold names in the benchmark
# suite (Nadia Kovalenko, Kenji Ito, Chidi Santos...), the classifier's
# ML confidence maxed out at 0.786, never once reaching the 0.85
# ML_HIGH_CONFIDENCE_OVERRIDE threshold that's the ONLY path a
# dictionary-absent candidate has to ACCEPTED (see decision_engine.py).
# Root cause: every existing positive synthetic example draws its name
# from the knowledge base and gets a DICTIONARY detection ~85% of the
# time (see _build_positive_examples/_build_context_cue_positive_examples),
# so the model has essentially learned "no dictionary hit = not a
# person" - a real, measured skew this asset-based distant-supervision
# approach was always going to have. These examples use algorithmically-
# generated pseudo-names (verified against the real dictionaries so
# they never accidentally collide) with ONLY regex + occasional spaCy
# evidence, exactly the shape a genuine unseen real name produces.
_PSEUDO_NAME_CONSONANTS = "bcdfghjklmnprstvwz"
_PSEUDO_NAME_VOWELS = "aeiou"


def _build_positive_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    # sorted(), not list() - KnowledgeBase's fields are frozensets, whose
    # iteration order depends on Python's per-process string hash
    # randomization (PYTHONHASHSEED). random.seed(42) alone does NOT make
    # this reproducible: random.choice() draws from whatever order the
    # set happens to iterate in, which differs run to run even with the
    # same seed. Found via real testing: re-running this script produced
    # a materially different synthetic dataset each time (same size,
    # same random.seed(42)), and the resulting trained models varied
    # wildly on real benchmark accuracy (accepted-only recall observed
    # ranging from ~9% to ~32% run to run) despite similar-looking
    # synthetic validation accuracy every time.
    first_names = sorted(kb.first_names)
    last_names = sorted(kb.last_names)
    examples = []

    n_full = int(count * 0.60)
    n_single = int(count * 0.20)
    n_chat_header = count - n_full - n_single

    for _ in range(n_full):
        first = random.choice(first_names).capitalize()
        last = random.choice(last_names).capitalize()
        bare_name = f"{first} {last}"

        # Match real RegexDetector behavior exactly: a "titled" match's
        # span INCLUDES the title token itself ("Dr. Suresh Kumar", not
        # "Suresh Kumar"), so titled and bare examples must differ in
        # BOTH token_count and has_title together, the same way they do
        # at inference - this is what a real train/inference skew bug
        # (found via direct testing) looks like when fixed.
        use_title = random.random() < 0.4
        if use_title:
            title = random.choice(TITLES_FOR_TEMPLATES)
            name = f"{title} {bare_name}"
        else:
            name = bare_name

        template = random.choice(POSITIVE_TEMPLATES)
        sentence = template.format(name=name)
        start = sentence.index(name)
        end = start + len(name)

        detections = [Detection(
            text=name, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.90 if use_title else 0.55,
            metadata={"pattern": "titled" if use_title else "bare"},
        )]
        if random.random() < 0.85:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.DICTIONARY, confidence=round(random.uniform(0.65, 0.85), 2),
                metadata={"pattern": "dictionary"},
            ))
        if random.random() < 0.6:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))

        examples.append(_to_example(sentence, name, start, end, detections, label=1, kind="positive_name"))

    # Single-token positives: only dictionary + optionally spaCy evidence
    # (no regex bare match is possible for a lone token) - this is the
    # exact real-world pattern a lone "Suresh" produces.
    for _ in range(n_single):
        first = random.choice(first_names).capitalize()
        template = random.choice(POSITIVE_SINGLE_TOKEN_TEMPLATES)
        sentence = template.format(name=first)
        start = sentence.index(first)
        end = start + len(first)

        detections = [Detection(
            text=first, start=start, end=end, page_index=0,
            detector=DetectorName.DICTIONARY, confidence=0.40,
            metadata={"pattern": "dictionary_single_token"},
        )]
        if random.random() < 0.5:
            detections.append(Detection(
                text=first, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))

        examples.append(_to_example(sentence, first, start, end, detections, label=1, kind="positive_single_token"))

    # Chat-header positives: {name} alone on its own line, immediately
    # followed by a newline then message text - see
    # POSITIVE_CHAT_HEADER_TEMPLATES' comment above for why this shape
    # matters. Half full name, half single-token, matching the mix real
    # chat exports actually show (a sender is sometimes saved as just a
    # first name). spaCy support is deliberately rare here (real
    # measured rate on bare chat-header lines, not the 50-60% used for
    # narrative-sentence positives above).
    n_chat_full = n_chat_header // 2
    n_chat_single = n_chat_header - n_chat_full

    for _ in range(n_chat_full):
        first = random.choice(first_names).capitalize()
        last = random.choice(last_names).capitalize()
        name = f"{first} {last}"
        template = random.choice(POSITIVE_CHAT_HEADER_TEMPLATES)
        sentence = template.format(name=name)
        start = sentence.index(name)
        end = start + len(name)

        detections = [Detection(
            text=name, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        if random.random() < 0.85:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.DICTIONARY, confidence=round(random.uniform(0.65, 0.85), 2),
                metadata={"pattern": "dictionary"},
            ))
        if random.random() < 0.08:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))

        examples.append(_to_example(sentence, name, start, end, detections, label=1, kind="positive_chat_header"))

    for _ in range(n_chat_single):
        first = random.choice(first_names).capitalize()
        template = random.choice(POSITIVE_CHAT_HEADER_TEMPLATES)
        sentence = template.format(name=first)
        start = sentence.index(first)
        end = start + len(first)

        detections = [Detection(
            text=first, start=start, end=end, page_index=0,
            detector=DetectorName.DICTIONARY, confidence=0.40,
            metadata={"pattern": "dictionary_single_token"},
        )]
        if random.random() < 0.08:
            detections.append(Detection(
                text=first, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))

        examples.append(_to_example(sentence, first, start, end, detections, label=1, kind="positive_chat_header_single_token"))

    return examples


def _build_negative_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    # sorted(), not list() - see _build_positive_examples' comment above.
    negative_phrases = sorted(kb.organizations) + sorted(kb.locations) + sorted(kb.campaigns) + \
        sorted(kb.blacklist) + GENERIC_CAPITALIZED_BIGRAMS
    first_names = sorted(kb.first_names)
    examples = []

    n_plain = int(count * 0.6)
    n_hard = int(count * 0.2)
    n_chat_header = count - n_plain - n_hard

    for _ in range(n_plain):
        phrase = random.choice(negative_phrases)
        display_phrase = " ".join(w.capitalize() for w in phrase.split())
        template = random.choice(NEGATIVE_TEMPLATES)
        sentence = template.format(name=display_phrase)
        start = sentence.index(display_phrase)
        end = start + len(display_phrase)

        detections = [Detection(
            text=display_phrase, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        if random.random() < 0.25:
            detections.append(Detection(
                text=display_phrase, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))

        examples.append(_to_example(sentence, display_phrase, start, end, detections, label=0, kind="negative_non_person"))

    # Hard negatives: real first name + organizational suffix. These
    # DO get dictionary hits (first_name_dict_hits=1), forcing the
    # model to use other features (last_name_dict_hits=0, char pattern,
    # detector agreement) rather than dict_hit_ratio alone.
    for _ in range(n_hard):
        first = random.choice(first_names).capitalize()
        suffix = random.choice(ORG_SUFFIXES)
        display_phrase = f"{first} {suffix}"
        template = random.choice(HARD_NEGATIVE_TEMPLATES)
        sentence = template.format(name=display_phrase)
        start = sentence.index(display_phrase)
        end = start + len(display_phrase)

        detections = [Detection(
            text=display_phrase, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        ), Detection(
            text=display_phrase, start=start, end=end, page_index=0,
            detector=DetectorName.DICTIONARY, confidence=0.68,
            metadata={"pattern": "dictionary"},
        )]

        examples.append(_to_example(sentence, display_phrase, start, end, detections, label=0, kind="negative_hard_org_with_real_first_name"))

    for _ in range(n_chat_header):
        phrase = random.choice(GENERIC_CAPITALIZED_BIGRAMS + CHAT_HEADER_NEGATIVE_PHRASES)
        template = random.choice(NEGATIVE_CHAT_HEADER_TEMPLATES)
        sentence = template.format(name=phrase)
        start = sentence.index(phrase)
        end = start + len(phrase)

        detections = [Detection(
            text=phrase, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        if random.random() < 0.08:
            detections.append(Detection(
                text=phrase, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))

        examples.append(_to_example(sentence, phrase, start, end, detections, label=0, kind="negative_chat_header"))

    return examples


def _build_context_cue_positive_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    """Real name + a preceding person-cue word ("According to", "signed
    by"...), nothing unusual following - reinforces
    preceding_context_is_person_cue=1 on the positive class."""
    first_names = sorted(kb.first_names)
    last_names = sorted(kb.last_names)
    examples = []
    for _ in range(count):
        first = random.choice(first_names).capitalize()
        last = random.choice(last_names).capitalize()
        name = f"{first} {last}"
        template = random.choice(CONTEXT_CUE_POSITIVE_TEMPLATES)
        sentence = template.format(name=name)
        start = sentence.index(name)
        end = start + len(name)

        detections = [Detection(
            text=name, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        if random.random() < 0.85:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.DICTIONARY, confidence=round(random.uniform(0.65, 0.85), 2),
                metadata={"pattern": "dictionary"},
            ))
        examples.append(_to_example(sentence, name, start, end, detections, label=1, kind="positive_context_cue"))
    return examples


def _build_context_cue_negative_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    """Real name immediately followed (OUTSIDE the candidate span) by a
    non-person cue word ("Ltd", "Division"...) - the "Marcus Webb Ltd"
    business-entity pattern. The candidate span covers only the
    name-shaped tokens, matching how a real RegexDetector/DictionaryDetector
    match would look here (the "Ltd"/"Division" suffix is not itself
    name-shaped, so it would never be part of the candidate span)."""
    first_names = sorted(kb.first_names)
    last_names = sorted(kb.last_names)
    cue_words = sorted(_NONPERSON_CONTEXT_CUES)
    examples = []
    for _ in range(count):
        first = random.choice(first_names).capitalize()
        last = random.choice(last_names).capitalize()
        name = f"{first} {last}"
        cue = random.choice(cue_words).capitalize()
        template = random.choice(CONTEXT_CUE_NEGATIVE_TEMPLATES)
        sentence = template.format(name=f"{name} {cue}")
        start = sentence.index(name)
        end = start + len(name)  # span excludes the trailing cue word

        detections = [Detection(
            text=name, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        if random.random() < 0.85:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.DICTIONARY, confidence=round(random.uniform(0.65, 0.85), 2),
                metadata={"pattern": "dictionary"},
            ))
        examples.append(_to_example(sentence, name, start, end, detections, label=0, kind="negative_context_cue"))
    return examples


def _build_unseen_name_positive_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    """See _PSEUDO_NAME_CONSONANTS' docstring above. Pseudo-names placed
    in the same strong-context templates already used for real
    dictionary names, but with regex (+ occasional spaCy) evidence only
    - no dictionary detection at all, matching how a genuine unseen real
    name is actually detected in production."""
    used: set[str] = set()

    def _fresh_pseudo_name() -> str:
        while True:
            syllables = random.randint(2, 3)
            token = "".join(
                random.choice(_PSEUDO_NAME_CONSONANTS) + random.choice(_PSEUDO_NAME_VOWELS)
                for _ in range(syllables)
            ).capitalize()
            if token.lower() in used:
                continue
            if kb.is_known_first_name(token) or kb.is_known_last_name(token):
                continue
            used.add(token.lower())
            return token

    templates_pool = POSITIVE_TEMPLATES + POSITIVE_CHAT_HEADER_TEMPLATES + CONTEXT_CUE_POSITIVE_TEMPLATES
    examples = []
    for _ in range(count):
        name = f"{_fresh_pseudo_name()} {_fresh_pseudo_name()}"
        template = random.choice(templates_pool)
        sentence = template.format(name=name)
        start = sentence.index(name)
        end = start + len(name)

        detections = [Detection(
            text=name, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        if random.random() < 0.5:
            detections.append(Detection(
                text=name, start=start, end=end, page_index=0,
                detector=DetectorName.SPACY, confidence=0.50,
                metadata={"pattern": "spacy_ner"},
            ))
        examples.append(_to_example(sentence, name, start, end, detections, label=1, kind="positive_unseen_name"))
    return examples


def _build_unseen_name_negative_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    """Negative counterpart to _build_unseen_name_positive_examples -
    without this, the added positives above could teach "no dictionary
    hit -> person" as a blanket rule instead of "no dictionary hit +
    real person-context -> person." Same pseudo-name generator, but
    placed in the existing non-person NEGATIVE_TEMPLATES/
    HARD_NEGATIVE_TEMPLATES shapes instead of person-context ones."""
    used: set[str] = set()

    def _fresh_pseudo_name() -> str:
        while True:
            syllables = random.randint(2, 3)
            token = "".join(
                random.choice(_PSEUDO_NAME_CONSONANTS) + random.choice(_PSEUDO_NAME_VOWELS)
                for _ in range(syllables)
            ).capitalize()
            if token.lower() in used:
                continue
            if kb.is_known_first_name(token) or kb.is_known_last_name(token):
                continue
            used.add(token.lower())
            return token

    templates_pool = NEGATIVE_TEMPLATES + HARD_NEGATIVE_TEMPLATES
    examples = []
    for _ in range(count):
        name = f"{_fresh_pseudo_name()} {_fresh_pseudo_name()}"
        template = random.choice(templates_pool)
        sentence = template.format(name=name)
        start = sentence.index(name)
        end = start + len(name)

        detections = [Detection(
            text=name, start=start, end=end, page_index=0,
            detector=DetectorName.REGEX, confidence=0.55,
            metadata={"pattern": "bare"},
        )]
        examples.append(_to_example(sentence, name, start, end, detections, label=0, kind="negative_unseen_name"))
    return examples


def _build_partial_dictionary_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    """Two-token spans where exactly ONE token is a dictionary name - the
    shape the pipeline produces for "Nadia Kovalenko" / "Devraj Bhatt"
    since the 2026-09-24 boundary fix keeps such names whole (bare regex
    span + a dictionary_single_token detection on the known token). No
    synthetic example had this shape before, so the classifier scored
    real ones near zero ("Femi Novak" 0.04) and they stalled in REVIEW.

    Positives: an unambiguous, non-common-word known first (or last) name
    beside a pseudo-name no list contains, in person contexts.
    Negatives, same detections: a known first name beside an ordinary
    English word ("Deep Tunnel", "Grace Period"-style), in non-person
    contexts - so the model must learn from the second word itself (e.g.
    common_word_ratio), not from the detection shape alone."""
    def usable(word: str) -> bool:
        return (word.isalpha() and word.isascii() and 3 <= len(word) <= 12
                and not kb.is_common_word(word) and not kb.is_ambiguous_first_name(word)
                and not kb.is_calendar_word(word) and not kb.is_stopword(word))

    first_names = [n for n in sorted(kb.first_names) if usable(n)]
    last_names = [n for n in sorted(kb.last_names) if usable(n)]
    common = [w for w in sorted(kb.common_words)
              if w.isalpha() and w.isascii() and 4 <= len(w) <= 10 and not kb.is_stopword(w)
              and not kb.is_known_first_name(w) and not kb.is_known_last_name(w)]
    used: set[str] = set()

    def pseudo() -> str:
        while True:
            token = "".join(random.choice(_PSEUDO_NAME_CONSONANTS) + random.choice(_PSEUDO_NAME_VOWELS)
                            for _ in range(random.randint(2, 3))).capitalize()
            if token.lower() in used or kb.is_known_first_name(token) or kb.is_known_last_name(token) \
                    or kb.is_common_word(token):
                continue
            used.add(token.lower())
            return token

    def example(tokens: list[str], known_index: int, templates: list[str], label: int, kind: str, spacy_p: float) -> dict:
        name = " ".join(tokens)
        sentence = random.choice(templates).format(name=name)
        start = sentence.index(name)
        end = start + len(name)
        known_start = start + (0 if known_index == 0 else len(tokens[0]) + 1)
        known = tokens[known_index]
        detections = [
            Detection(text=name, start=start, end=end, page_index=0, detector=DetectorName.REGEX,
                      confidence=0.55, metadata={"pattern": "bare"}),
            Detection(text=known, start=known_start, end=known_start + len(known), page_index=0,
                      detector=DetectorName.DICTIONARY, confidence=0.40,
                      metadata={"pattern": "dictionary_single_token"}),
        ]
        if random.random() < spacy_p:
            detections.append(Detection(text=name, start=start, end=end, page_index=0, detector=DetectorName.SPACY,
                                        confidence=0.50, metadata={"pattern": "spacy_ner"}))
        return _to_example(sentence, name, start, end, detections, label=label, kind=kind)

    positive_templates = POSITIVE_TEMPLATES + POSITIVE_CHAT_HEADER_TEMPLATES + CONTEXT_CUE_POSITIVE_TEMPLATES
    negative_templates = NEGATIVE_TEMPLATES + HARD_NEGATIVE_TEMPLATES + NEGATIVE_CHAT_HEADER_TEMPLATES
    examples = []
    for i in range(count):
        if i % 2 == 0:
            examples.append(example([random.choice(first_names).capitalize(), pseudo()], 0,
                                    positive_templates, 1, "positive_partial_dictionary", 0.4))
        else:
            examples.append(example([pseudo(), random.choice(last_names).capitalize()], 1,
                                    positive_templates, 1, "positive_partial_dictionary", 0.4))
        examples.append(example([random.choice(first_names).capitalize(), random.choice(common).capitalize()], 0,
                                negative_templates, 0, "negative_partial_dictionary_common_word", 0.25))
    return examples


def _to_example(sentence: str, text: str, start: int, end: int, detections: list[Detection], label: int, kind: str) -> dict:
    candidate = Candidate.new(
        text=text, normalized_text=text, start=start, end=end,
        page_index=0, source_detections=tuple(detections),
    )
    candidate_result = CandidateResult(candidate=candidate, state=CandidateState())
    kb = _KB_SINGLETON
    features = extract_features(candidate_result, sentence, kb)
    return {"text": text, "kind": kind, "label": label, "features": features.as_list(), "context_text": sentence}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced synthetic training data")
    parser.add_argument("--scale", type=int, default=1,
                        help="Multiply every balanced example family by this amount (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for a reproducible corpus (default: 42)")
    parser.add_argument("--output", type=Path,
                        default=BASE_DIR / "datasets" / "training" / "synthetic_training_data.jsonl",
                        help="JSONL destination for the generated training data")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be at least 1")

    random.seed(args.seed)
    global _KB_SINGLETON
    assets_dir = BASE_DIR / "assets"
    _KB_SINGLETON = KnowledgeBase.load(assets_dir)

    n_positive = 1200 * args.scale
    n_negative = 1200 * args.scale
    n_context_cue = 150 * args.scale  # each class, see CONTEXT_CUE_*_TEMPLATES' docstring
    n_unseen_name = 600 * args.scale  # see _PSEUDO_NAME_CONSONANTS' docstring. Raised
    # from 200 to 600 on 2026-09-18: the prior round (README's
    # "unseen_name ML confidence ceiling" entry) moved max confidence on
    # real unseen gold mentions 0.786 -> 0.838, still short of
    # ML_HIGH_CONFIDENCE_OVERRIDE's 0.85 by 0.012 - close enough that
    # more of the same targeted reinforcement (not a threshold change,
    # per that entry's explicit reasoning) was worth trying before
    # concluding the gap needs a different mechanism entirely.
    n_partial_dictionary = 300 * args.scale  # each class - see _build_partial_dictionary_examples

    positives = _build_positive_examples(_KB_SINGLETON, n_positive)
    positives += _build_context_cue_positive_examples(_KB_SINGLETON, n_context_cue)
    positives += _build_unseen_name_positive_examples(_KB_SINGLETON, n_unseen_name)
    negatives = _build_negative_examples(_KB_SINGLETON, n_negative)
    negatives += _build_context_cue_negative_examples(_KB_SINGLETON, n_context_cue)
    negatives += _build_unseen_name_negative_examples(_KB_SINGLETON, n_unseen_name)
    # Built LAST so every example above is drawn from the same random
    # stream as before this builder existed (only the shuffle order moves).
    partial = _build_partial_dictionary_examples(_KB_SINGLETON, n_partial_dictionary)
    positives += [e for e in partial if e["label"] == 1]
    negatives += [e for e in partial if e["label"] == 0]

    all_examples = positives + negatives
    random.shuffle(all_examples)

    out_path = args.output
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} synthetic examples to {out_path} "
          f"(scale={args.scale}, seed={args.seed})")
    print(f"  positive (person):     {len(positives)}")
    print(f"  negative (non-person): {len(negatives)}")                                                   


_KB_SINGLETON: KnowledgeBase | None = None

if __name__ == "__main__":
    main()
