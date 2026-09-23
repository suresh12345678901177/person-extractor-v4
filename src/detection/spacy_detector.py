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

from typing import Iterator

from src.core.models import Detection, DetectorName
from src.detection.base_detector import BaseDetector
from src.utils.logger import get_logger

logger = get_logger("detection.spacy_detector")


def iter_line_bounded_chunks(text: str, max_chars: int) -> Iterator[tuple[int, str]]:
    """Yield (offset, chunk) pairs covering the whole of `text`, each at
    most max_chars long, split only at line boundaries so a chunk
    boundary can never fall inside a token/name - the same "never split
    mid-line" stance this file already takes for multi-line entities
    (see the cross-newline splitting logic below) and regex_detector.py
    takes for its own matches. A single line itself longer than
    max_chars (e.g. a one-line JSON/hex blob some Android dumps contain)
    is hard-split as a last resort - it could never be kept whole in one
    spaCy call either way, and this only ever improves on the
    alternative (zero detections for the whole document).

    Concatenating every yielded chunk reconstructs `text` exactly, so
    `offset + local_position` always maps back to the correct position
    in the original text.
    """
    if len(text) <= max_chars:
        yield 0, text
        return

    offset = 0
    buf_start = 0
    buf_len = 0
    for line in text.splitlines(keepends=True):
        if len(line) > max_chars:
            if buf_len:
                yield buf_start, text[buf_start:offset]
                buf_start, buf_len = offset, 0
            sub = 0
            while sub < len(line):
                yield offset + sub, line[sub:sub + max_chars]
                sub += max_chars
            offset += len(line)
            buf_start = offset
            continue
        if buf_len + len(line) > max_chars:
            yield buf_start, text[buf_start:offset]
            buf_start, buf_len = offset, 0
        buf_len += len(line)
        offset += len(line)
    if buf_len:
        yield buf_start, text[buf_start:offset]


class SpacyDetector(BaseDetector):
    name = DetectorName.SPACY

    CONFIDENCE = 0.50

    # Comfortably under spaCy's 1,000,000-char hard nlp.max_length
    # ceiling (with real margin for its own per-call memory overhead),
    # so a document of ANY size gets processed in bounded-size pieces
    # instead of the whole call raising "exceeds maximum" - which
    # detect() previously caught via a blanket except-Exception and
    # silently returned zero detections for. Confirmed via
    # logs/pipeline.log: this was happening 142 times on this project's
    # real (multi-MB Android dump) casework files, meaning those
    # documents were getting ZERO spaCy corroboration, not just running
    # slow. Line-bounded chunking (see iter_line_bounded_chunks above)
    # fixes the coverage gap; disabling tagger/parser below (never read
    # anywhere in this codebase - only doc.ents is used) cuts the
    # per-chunk cost by roughly a quarter on top of that, with byte-for-
    # byte identical entity output (verified).
    MAX_CHUNK_CHARS = 400_000

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
                "en_core_web_sm",
                # lemmatizer/attribute_ruler: never used. tagger/parser:
                # also never used (grepped the whole codebase - nothing
                # reads .pos_/.dep_/.tag_/doc.sents) - only doc.ents
                # (NER) is read anywhere. Disabling both saves real,
                # measured time (~26% faster on a 300K-char sample,
                # identical entity output) on every single call, which
                # matters most on exactly the huge documents this
                # project's real casework is full of.
                disable=["lemmatizer", "attribute_ruler", "tagger", "parser"],
            )
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run: "
                "python -m spacy download en_core_web_sm"
            ) from exc

    def detect(self, text: str, page_index: int) -> list[Detection]:
        if not text.strip() or self._nlp is None:
            return []

        detections: list[Detection] = []
        for chunk_offset, chunk_text in iter_line_bounded_chunks(text, self.MAX_CHUNK_CHARS):
            if not chunk_text.strip():
                continue
            try:
                doc = self._nlp(chunk_text)
            except Exception:
                logger.exception(
                    "spaCy failed to process page %d (chunk at offset %d, %d chars)",
                    page_index, chunk_offset, len(chunk_text),
                )
                continue
            detections.extend(self._entities_to_detections(doc, page_index, chunk_offset))
        return detections

    @classmethod
    def _entities_to_detections(cls, doc, page_index: int, chunk_offset: int) -> list[Detection]:
        detections: list[Detection] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            if "\n" not in ent.text:
                detections.append(
                    Detection(
                        text=ent.text,
                        start=chunk_offset + ent.start_char,
                        end=chunk_offset + ent.end_char,
                        page_index=page_index,
                        detector=cls.name,
                        confidence=cls.CONFIDENCE,
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
                        start=chunk_offset + seg_start,
                        end=chunk_offset + seg_end,
                        page_index=page_index,
                        detector=cls.name,
                        confidence=cls.CONFIDENCE,
                        metadata={"pattern": "spacy_ner", "spacy_label": ent.label_},
                    )
                )
        return detections
