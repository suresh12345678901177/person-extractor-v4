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

import re
from typing import Iterator

from src.core.models import Detection, DetectorName
from src.detection.base_detector import BaseDetector
from src.utils.logger import get_logger

logger = get_logger("detection.spacy_detector")

_WHITESPACE_TOKEN_RE = re.compile(r"\S+")


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

    # Loaded pipelines by model name, shared across instances (model load is
    # slow). Keyed by name since 2026-09-25: model_name used to be accepted
    # and then ignored - "en_core_web_sm" was hard-coded in spacy.load.
    _nlps: dict = {}

    def __init__(self, model_name: str = "en_core_web_sm", n_process: int = 1) -> None:
        self.model_name = model_name
        # See config.py's detection.spacy_processes for when to raise this.
        self.n_process = max(1, int(n_process))
        self._nlp = self._load(model_name)

    @property
    def model_version(self) -> str:
        """e.g. 'en_core_web_sm-3.8.0' - stamped into run provenance."""
        meta = self._nlp.meta
        return f"{meta.get('lang', '')}_{meta.get('name', '')}-{meta.get('version', '')}"

    @classmethod
    def _load(cls, model_name: str):
        if model_name in cls._nlps:
            return cls._nlps[model_name]
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is required for SpacyDetector. Install it with: "
                f"pip install spacy && python -m spacy download {model_name}"
            ) from exc

        try:
            cls._nlps[model_name] = spacy.load(
                model_name,
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
                f"spaCy model '{model_name}' is not installed. Run: "
                f"python -m spacy download {model_name}"
            ) from exc
        return cls._nlps[model_name]

    def detect(self, text: str, page_index: int) -> list[Detection]:
        if not text.strip() or self._nlp is None:
            return []

        chunks = [
            (offset, chunk) for offset, chunk in iter_line_bounded_chunks(text, self.MAX_CHUNK_CHARS)
            if chunk.strip()
        ]

        if self.n_process > 1 and len(chunks) > 1:
            parallel = self._detect_parallel(chunks, page_index)
            if parallel is not None:
                return parallel

        detections: list[Detection] = []
        for chunk_offset, chunk_text in chunks:
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

    def _detect_parallel(self, chunks: list[tuple[int, str]], page_index: int) -> list[Detection] | None:
        """Same chunks, same model, spread over worker processes via
        nlp.pipe(n_process=...). Chunk boundaries are unchanged from the
        sequential path, so entity output is identical (verified on a
        5.4MB file: 56,791 PERSON entities, byte-identical spans). Returns
        None on any failure so detect() falls back to the sequential,
        per-chunk-error-tolerant path rather than losing all detections."""
        try:
            docs = self._nlp.pipe(
                (chunk for _, chunk in chunks),
                n_process=min(self.n_process, len(chunks)),
                batch_size=1,
            )
            detections: list[Detection] = []
            for (chunk_offset, _), doc in zip(chunks, docs):
                detections.extend(self._entities_to_detections(doc, page_index, chunk_offset))
            return detections
        except Exception:
            logger.exception(
                "Parallel spaCy (n_process=%d) failed on page %d - retrying sequentially",
                self.n_process, page_index,
            )
            return None

    @classmethod
    def _entities_to_detections(cls, doc, page_index: int, chunk_offset: int) -> list[Detection]:
        detections: list[Detection] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
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
            #
            # The same holds for an entity running across a phone number
            # or any other digit-bearing token (2026-09-28): on one-line
            # call-log/chat dumps ("919812345670 Farhan Vora 919812345671
            # Meera Iyengar Hi") spaCy tagged "Farhan Vora 919812345671
            # Meera Iyengar Hi" as ONE person. CandidateFactory then merged
            # every detection it overlapped into one cluster that kept only
            # "Meera Iyengar" - "Farhan Vora" vanished from that spot. No
            # person's name contains a digit, so such tokens split the
            # entity (and are dropped) exactly like a line break does.
            for seg_start, seg_end in cls._name_segments(ent.text):
                detections.append(
                    Detection(
                        text=ent.text[seg_start:seg_end],
                        start=chunk_offset + ent.start_char + seg_start,
                        end=chunk_offset + ent.start_char + seg_end,
                        page_index=page_index,
                        detector=cls.name,
                        confidence=cls.CONFIDENCE,
                        metadata={"pattern": "spacy_ner", "spacy_label": ent.label_},
                    )
                )
        return detections

    @staticmethod
    def _name_segments(text: str) -> list[tuple[int, int]]:
        """(start, end) offsets within `text` of each run of consecutive
        whitespace-separated tokens that crosses no line break and contains
        no token with a digit in it. An entity with neither comes back as
        one segment, identical to the entity text."""
        segments: list[tuple[int, int]] = []
        run_start = run_end = None
        for m in _WHITESPACE_TOKEN_RE.finditer(text):
            if any(ch.isdigit() for ch in m.group()):
                if run_start is not None:
                    segments.append((run_start, run_end))
                run_start = run_end = None
                continue
            if run_start is not None and "\n" in text[run_end:m.start()]:
                segments.append((run_start, run_end))
                run_start = None
            if run_start is None:
                run_start = m.start()
            run_end = m.end()
        if run_start is not None:
            segments.append((run_start, run_end))
        return segments
