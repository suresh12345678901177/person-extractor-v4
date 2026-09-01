"""
src.core.models
================
Core data models for PERSON_EXTRACTOR_V4.

Naming and structure follow the PERSON_EXTRACTOR_V4 Architecture
Blueprint's "Core Models" section exactly: Candidate, CandidateState,
CandidateResult, Detection, Document, Page, DocumentResult,
PipelineContext, PipelineStatistics, ExtractionResult, ValidationResult,
FeatureVector.

Design rules (per blueprint "Design Principles"):
- Immutable Core Models: Candidate, Detection, Document, Page, and
  FeatureVector are frozen + slots=True. They are never mutated after
  creation.
- CandidateState is the one deliberately mutable object: it accumulates
  evidence as a candidate flows through the pipeline.
- CandidateResult pairs an immutable Candidate with its mutable
  CandidateState. From candidate-creation onward, every pipeline stage
  operates ONLY on CandidateResult, never on Candidate directly.
- Every stage that can fail returns a structured *Result object rather
  than raising or returning a bare bool/None ("no silent failure").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ------------------------------------------------------------------ #
# Enums
# ------------------------------------------------------------------ #

class SourceFormat(str, Enum):
    """Supported input formats. V4 scope is TXT-only; the enum keeps the
    door open for future formats without touching any existing code."""
    TXT = "txt"


class DetectorName(str, Enum):
    REGEX = "regex"
    DICTIONARY = "dictionary"
    SPACY = "spacy"


class Decision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW = "review"


# ------------------------------------------------------------------ #
# Raw input models (post-reading, pre-detection)
# ------------------------------------------------------------------ #

@dataclass(frozen=True, slots=True)
class Page:
    """A single page/segment of a source document. TXT files are always
    a single Page (index 0); the field exists so future multi-page
    formats need no model changes."""
    index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Document:
    """The normalized representation of any input file, post-reading."""
    source_path: str
    source_format: SourceFormat
    pages: tuple[Page, ...]
    full_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(slots=True)
class DocumentResult:
    """Structured, never-raise result of the reading stage for one file."""
    success: bool
    document: Document | None = None
    source_path: str = ""
    error: str | None = None


# ------------------------------------------------------------------ #
# Detection -> Candidate models
# ------------------------------------------------------------------ #

@dataclass(frozen=True, slots=True)
class Detection:
    """Raw output emitted by a single Detector for a single span of text."""
    text: str
    start: int
    end: int
    page_index: int
    detector: DetectorName
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextLocation:
    """Human-readable location of a span within a TXT file, computed by
    the preprocessing line-indexer. Char offsets are precise but not
    human-friendly on their own; line/column make the "location" output
    requirement legible without opening the file in a hex editor."""
    char_start: int
    char_end: int
    line_number: int      # 1-indexed
    column_number: int    # 1-indexed, column of char_start on its line

    def __str__(self) -> str:
        return f"line {self.line_number}, col {self.column_number} (chars {self.char_start}-{self.char_end})"


@dataclass(frozen=True, slots=True)
class Candidate:
    """
    Immutable candidate person-name span, produced by merging one or more
    Detection objects that refer to the same underlying span of text.
    """
    candidate_id: str
    text: str
    normalized_text: str
    start: int
    end: int
    page_index: int
    source_detections: tuple[Detection, ...]
    location: TextLocation | None = None

    @staticmethod
    def new(
        text: str,
        normalized_text: str,
        start: int,
        end: int,
        page_index: int,
        source_detections: tuple[Detection, ...],
        location: TextLocation | None = None,
    ) -> "Candidate":
        return Candidate(
            candidate_id=str(uuid.uuid4()),
            text=text,
            normalized_text=normalized_text,
            start=start,
            end=end,
            page_index=page_index,
            source_detections=source_detections,
            location=location,
        )

    @property
    def detector_names(self) -> tuple[str, ...]:
        return tuple(sorted({d.detector.value for d in self.source_detections}))


@dataclass(slots=True)
class EvidenceItem:
    """A single scored piece of evidence contributed by a pipeline stage.
    This is the backbone of the "why is this a name" explainability
    requirement: every point added or subtracted from a candidate's score
    is recorded here with a human-readable label."""
    source: str      # e.g. "regex", "dictionary_validator", "ml_classifier"
    label: str        # human-readable description, e.g. "Title match"
    score: float       # signed contribution to the final confidence


@dataclass(slots=True)
class ValidationResult:
    """Result returned by every Validator. NEVER a bare boolean, always
    this structured object so downstream stages (and explainability) can
    see why. Named ValidationResult per the blueprint (V3 called this
    ValidatorResult; V4 standardizes on the blueprint's name)."""
    validator_name: str
    passed: bool
    score: float = 0.0
    severity: str = "soft"   # "soft" (affects score) | "hard" (auto-reject)
    message: str = ""


@dataclass(slots=True)
class FeatureVector:
    """Fixed-order numeric/categorical feature vector for the ML
    classifier, produced by src.features.feature_extractor. Keeping this
    as an explicit named object (rather than a bare list/array) means
    every feature is documented and the training script and inference
    path can never silently drift out of sync with each other."""
    token_count: int
    char_length: int
    has_title: int                 # 0/1
    has_honorific: int             # 0/1
    first_name_dict_hits: int
    last_name_dict_hits: int
    dict_hit_ratio: float
    capitalized_ratio: float       # fraction of tokens that are capitalized
    initial_token_count: int       # tokens like "S." or "R"
    regex_confidence: float        # 0 if no regex detection
    dictionary_confidence: float   # 0 if no dictionary detection
    spacy_confidence: float        # 0 if no spaCy detection
    detector_count: int            # how many distinct detectors fired
    preceding_word_is_stopword: int  # 0/1
    following_char_is_punct: int     # 0/1
    occurs_in_quotes: int             # 0/1 - inside a quoted string

    def as_list(self) -> list[float]:
        """Fixed-order numeric encoding for the classifier. Order MUST
        match FEATURE_NAMES in src.features.feature_vector exactly."""
        return [
            float(self.token_count),
            float(self.char_length),
            float(self.has_title),
            float(self.has_honorific),
            float(self.first_name_dict_hits),
            float(self.last_name_dict_hits),
            float(self.dict_hit_ratio),
            float(self.capitalized_ratio),
            float(self.initial_token_count),
            float(self.regex_confidence),
            float(self.dictionary_confidence),
            float(self.spacy_confidence),
            float(self.detector_count),
            float(self.preceding_word_is_stopword),
            float(self.following_char_is_punct),
            float(self.occurs_in_quotes),
        ]


@dataclass(slots=True)
class ClassifierResult:
    """Result returned by the ML classifier's predict step for one candidate."""
    label: int
    probability: float
    model_name: str
    model_version: str = "unknown"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class CandidateState:
    """
    Mutable state accumulated for a Candidate as it flows through the
    pipeline: evidence, validation results, ML score, and final decision.
    """
    evidence: list[EvidenceItem] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    feature_vector: FeatureVector | None = None
    classifier_result: ClassifierResult | None = None
    final_confidence: float = 0.0
    decision: Decision | None = None
    rejection_reason: str | None = None
    # How many times this candidate's exact normalized text recurs
    # elsewhere in the document (computed once by CorroborationValidator,
    # reused by DecisionEngine) - lets the ML-high-confidence
    # knowledge-corroboration path require real repetition instead of
    # trusting a single occurrence's score alone. See decision_engine.py's
    # ML_HIGH_CONFIDENCE_OVERRIDE handling for why this matters: a
    # single-occurrence candidate with zero dictionary/title support can
    # still score an ML probability high enough to otherwise satisfy that
    # gate on shape alone.
    repetition_count: int = 0

    def add_evidence(self, source: str, label: str, score: float) -> None:
        self.evidence.append(EvidenceItem(source=source, label=label, score=score))

    @property
    def total_evidence_score(self) -> float:
        return sum(e.score for e in self.evidence)

    @property
    def is_hard_rejected(self) -> bool:
        return any(not vr.passed and vr.severity == "hard" for vr in self.validation_results)


@dataclass(slots=True)
class CandidateResult:
    """
    Combines an immutable Candidate with its mutable CandidateState.
    From candidate-creation onward, every pipeline stage should operate
    only on CandidateResult objects (never Candidate directly).
    """
    candidate: Candidate
    state: CandidateState = field(default_factory=CandidateState)

    @property
    def text(self) -> str:
        return self.candidate.text

    def explain(self) -> str:
        """Human-readable "why is this a name" explainability trace."""
        lines = [f"Candidate: {self.candidate.text}", "Evidence:"]
        for e in self.state.evidence:
            sign = "+" if e.score >= 0 else ""
            lines.append(f"  {e.label:<40} {sign}{e.score:.2f}  ({e.source})")
        lines.append(f"Final Score: {self.state.final_confidence:.2f}")
        lines.append(f"Decision: {self.state.decision.value if self.state.decision else 'pending'}")
        if self.state.rejection_reason:
            lines.append(f"Reason: {self.state.rejection_reason}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Aggregated (post-pipeline) result: one entry per unique person
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class AggregatedPerson:
    """
    One row per UNIQUE person name (grouped by normalized text), combining
    every mention found in the document. This directly implements the
    "how many times found" and "location(s)" output requirements without
    losing per-mention explainability - `mentions` still holds every
    individual CandidateResult with its own evidence trace.

    Despite the name, nothing here is actually person-specific (no field
    is anything but a generic "grouped entity with N mentions"). Left
    as-is rather than renamed to AggregatedEntity during the OOP extractor
    refactor (2026-08-26, see src/extraction/) - that rename would ripple
    into cli.py, all three exporters, feedback_store.py, and evaluator.py
    for zero correctness benefit while only one extraction type exists.
    Revisit once a second Extractor (e.g. phone numbers) actually needs
    to decide whether to reuse this or define its own result shape - a
    real decision is better made with a second use case in hand than
    guessed at now.
    """
    normalized_text: str
    display_text: str                    # most common surface form
    mentions: list[CandidateResult] = field(default_factory=list)

    @property
    def occurrence_count(self) -> int:
        return len(self.mentions)

    @property
    def locations(self) -> list[TextLocation]:
        return [m.candidate.location for m in self.mentions if m.candidate.location is not None]

    @property
    def best_confidence(self) -> float:
        return max((m.state.final_confidence for m in self.mentions), default=0.0)

    @property
    def decision(self) -> Decision | None:
        """A person is ACCEPTED if at least one mention was accepted -
        repeated weak mentions of an already-confirmed real name should
        not be suppressed just because one occurrence had less context."""
        decisions = {m.state.decision for m in self.mentions}
        if Decision.ACCEPTED in decisions:
            return Decision.ACCEPTED
        if Decision.REVIEW in decisions:
            return Decision.REVIEW
        return Decision.REJECTED


# ------------------------------------------------------------------ #
# Structured, timed pipeline results
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class StageTiming:
    """Time taken for a single named pipeline stage - the "time taken
    for the task" output requirement, broken down per stage rather than
    a single opaque total."""
    stage_name: str
    seconds: float


@dataclass(slots=True)
class PipelineStatistics:
    """Aggregated counters/timings for one pipeline run."""
    documents_processed: int = 0
    documents_failed: int = 0
    raw_detections: int = 0
    candidates_generated: int = 0
    unique_persons_found: int = 0
    candidates_accepted: int = 0
    candidates_review: int = 0
    candidates_rejected: int = 0
    stage_timings: list[StageTiming] = field(default_factory=list)
    total_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "raw_detections": self.raw_detections,
            "candidates_generated": self.candidates_generated,
            "unique_persons_found": self.unique_persons_found,
            "candidates_accepted": self.candidates_accepted,
            "candidates_review": self.candidates_review,
            "candidates_rejected": self.candidates_rejected,
            "stage_timings": {s.stage_name: round(s.seconds, 4) for s in self.stage_timings},
            "total_seconds": round(self.total_seconds, 4),
        }


@dataclass(slots=True)
class ExtractionResult:
    """Final, structured result of running the full pipeline on one document."""
    success: bool
    source_path: str
    persons: tuple[AggregatedPerson, ...] = ()      # ACCEPTED, grouped
    review_persons: tuple[AggregatedPerson, ...] = ()  # REVIEW, grouped
    rejected: tuple[CandidateResult, ...] = ()          # ungrouped (diagnostic)
    # Spot-check-only visibility net: candidates hard-rejected SPECIFICALLY
    # by CorroborationValidator (bare shape-match, zero dictionary/title/
    # spaCy support) - these never reach REVIEW or the feedback queue at
    # all under normal operation, so without this list a genuine person
    # meeting that description is completely invisible. Kept separate
    # from persons/review_persons: does NOT affect ACCEPTED/REVIEW counts,
    # CSV/JSON exports, or the feedback-labeling queue - purely a manual
    # spot-check list for report.txt/terminal output.
    low_confidence_persons: tuple[AggregatedPerson, ...] = ()
    error: str | None = None
    statistics: PipelineStatistics = field(default_factory=PipelineStatistics)
    model_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineContext:
    """
    Carries configuration and shared services through a pipeline run.
    Avoids global state: every stage receives context explicitly.
    """
    config: dict[str, Any]
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra: dict[str, Any] = field(default_factory=dict)
