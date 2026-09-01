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
Produces:
    datasets/training/synthetic_training_data.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.models import Candidate, CandidateResult, CandidateState, Detection, DetectorName
from src.features.feature_extractor import extract_features
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

    n_full = int(count * 0.75)
    n_single = count - n_full

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

    return examples


def _build_negative_examples(kb: KnowledgeBase, count: int) -> list[dict]:
    # sorted(), not list() - see _build_positive_examples' comment above.
    negative_phrases = sorted(kb.organizations) + sorted(kb.locations) + sorted(kb.campaigns) + \
        sorted(kb.blacklist) + GENERIC_CAPITALIZED_BIGRAMS
    first_names = sorted(kb.first_names)
    examples = []

    n_plain = int(count * 0.7)
    n_hard = count - n_plain

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

    return examples


def _to_example(sentence: str, text: str, start: int, end: int, detections: list[Detection], label: int, kind: str) -> dict:
    candidate = Candidate.new(
        text=text, normalized_text=text, start=start, end=end,
        page_index=0, source_detections=tuple(detections),
    )
    candidate_result = CandidateResult(candidate=candidate, state=CandidateState())
    kb = _KB_SINGLETON
    features = extract_features(candidate_result, sentence, kb)
    return {"text": text, "kind": kind, "label": label, "features": features.as_list()}


def main() -> None:
    global _KB_SINGLETON
    assets_dir = BASE_DIR / "assets"
    _KB_SINGLETON = KnowledgeBase.load(assets_dir)

    n_positive = 1200
    n_negative = 1200

    positives = _build_positive_examples(_KB_SINGLETON, n_positive)
    negatives = _build_negative_examples(_KB_SINGLETON, n_negative)

    all_examples = positives + negatives
    random.shuffle(all_examples)

    out_dir = BASE_DIR / "datasets" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_training_data.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} synthetic examples to {out_path}")
    print(f"  positive (person):     {len(positives)}")
    print(f"  negative (non-person): {len(negatives)}")                                                   


_KB_SINGLETON: KnowledgeBase | None = None

if __name__ == "__main__":
    main()
