"""
src.detection.spacy_detector
==============================
spaCy-based NER detector. Uses spaCy's statistical PERSON entity
recognition, which - unlike the regex/dictionary detectors - can catch a
name from a single token using surrounding sentence context (e.g.
"Suresh claimed he overheard...").

Deliberately given MODERATE confidence, not high: this model is not
tuned for Indian/regional names and occasionally mislabels an unrelated
capitalized word as PERSON (observed in testing: "Ignore", "Madam"). It
is one corroborating signal among several, combined by the Decision
Engine with rule-based evidence, and can never by itself push a
candidate past a hard validator rejection - this is the core "rules are
the gate, ML corroborates" design principle behind V4's no-FP goal.
"""

from __future__ import annotations

from src.core.models import Detection, DetectorName
from src.detection.base_detector import BaseDetector
from src.utils.logger import get_logger

logger = get_logger("detection.spacy_detector")


class SpacyDetector(BaseDetector):
    name = DetectorName.SPACY

    CONFIDENCE = 0.50

    _nlp = None  # loaded lazily and shared across instances (model load is slow)

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._nlp is not None:
            return
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is required for SpacyDetector. Install it with: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            ) from exc

        try:
            cls._nlp = spacy.load(
                "en_core_web_sm", disable=["lemmatizer", "attribute_ruler"]
            )
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run: "
                "python -m spacy download en_core_web_sm"
            ) from exc

    def detect(self, text: str, page_index: int) -> list[Detection]:
        if not text.strip() or self._nlp is None:
            return []

        try:
            doc = self._nlp(text)
        except Exception:
            logger.exception("spaCy failed to process page %d", page_index)
            return []

        detections: list[Detection] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            if "\n" not in ent.text:
                detections.append(
                    Detection(
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        page_index=page_index,
                        detector=self.name,
                        confidence=self.CONFIDENCE,
                        metadata={"pattern": "spacy_ner", "spacy_label": ent.label_},
                    )
                )
                continue

            # spaCy's NER has no hard constraint against an entity
            # crossing a line break, unlike the regex/dictionary
            # detectors which both explicitly forbid it (see
            # regex_detector.py's module docstring). A crossed-line
            # entity is frequently NOT "one real name plus trailing
            # junk" - on chat exports where a speaker-header name sits
            # directly above the next message ("Marco Silvestri\nDragan
            # Kovac already confirmed..."), spaCy merges TWO different
            # real people into one entity. An earlier version of this
            # fix truncated to just the text before the first newline,
            # which rescued only the FIRST name and silently discarded
            # every other real name crossed by the same merged entity -
            # the more names a merge crossed, the more got dropped
            # (this is what was costing "Dragan Kovac" its spaCy
            # corroboration on the majority of its real mentions).
            # Emitting one detection per non-empty line-segment instead
            # keeps every real name in the merge; a genuine trailing
            # label word ("Alias", "Phone") still can't pass the
            # decision gate on its own since it has no dictionary/title
            # support - it stays low-confidence, not missing evidence
            # for a different real person.
            cursor = ent.start_char
            for line in ent.text.split("\n"):
                line_start = cursor
                cursor += len(line) + 1  # +1 for the '\n' consumed by split
                stripped = line.strip()
                if not stripped:
                    continue
                seg_start = line_start + (len(line) - len(line.lstrip()))
                seg_end = seg_start + len(stripped)
                detections.append(
                    Detection(
                        text=stripped,
                        start=seg_start,
                        end=seg_end,
                        page_index=page_index,
                        detector=self.name,
                        confidence=self.CONFIDENCE,
                        metadata={"pattern": "spacy_ner", "spacy_label": ent.label_},
                    )
                )
        return detections
