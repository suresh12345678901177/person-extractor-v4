"""
src.extraction.person_extractor
==================================
The person-name extraction pipeline, as an `Extractor`. This is a
behavior-preserving move, not a rewrite: every stage here ran exactly
this way inline in `Pipeline.run()` before the OOP extractor refactor
(2026-08-26) - moved out so the person-specific machinery (knowledge
base, detectors, validators, decision engine, ML classifier) lives in
one self-contained, independently swappable unit, and a future
extraction type (phone numbers, ID numbers, ...) can be added as a
sibling class implementing the same `Extractor` interface without
touching any of this.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from src.candidate.boundary_refiner import (
    refine_boundaries,
    split_double_name_candidates,
    trim_dictionary_confirmed_organization_edge,
    trim_stopword_edges,
)
from src.candidate.candidate_aggregator import aggregate_by_surface_form
from src.candidate.candidate_factory import CandidateFactory
from src.candidate.name_propagation import (
    DOCUMENT_NAME_PATTERN,
    document_name_detection,
    propagatable_first_names,
    review_candidates_to_promote,
    undetected_occurrences,
)
from src.classification.model_registry import load_classifier
from src.core.models import (
    IDENTITY_RESOLUTION_CAVEAT,
    CandidateResult,
    Decision,
    Detection,
    DetectorName,
    ExtractionResult,
    PipelineStatistics,
    SourceFormat,
    STRUCTURED_FORMATS,
)
from src.decision.decision_engine import DecisionEngine
from src.detection.detector_manager import DetectorManager
from src.extraction.base_extractor import Extractor
from src.features.feature_extractor import extract_features
from src.feedback.feedback_store import FeedbackStore
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.language_filter import check_english
from src.preprocessing.line_indexer import LineIndex
from src.utils.logger import get_logger
from src.utils.provenance import build_run_provenance
from src.utils.timing import timed_stage
from src.validation.validation_pipeline import ValidationPipeline

logger = get_logger("extraction.person_extractor")


class PersonExtractor(Extractor):
    def __init__(self, config: dict[str, Any], base_dir: str | Path) -> None:
        super().__init__(config, base_dir)

        self.knowledge_base = KnowledgeBase.load(self.base_dir / "assets")
        self.detector_manager = DetectorManager(config, self.knowledge_base)

        # strict_corroboration=False is a comparison/testing knob (see
        # cli.py's --loose-gate) - OFF by default, meaning the documented
        # precision-first gate (dictionary/title/high-confidence-ML
        # required for ACCEPTED) stays exactly as designed unless
        # explicitly overridden.
        decision_config = config.get("decision", {})
        strict_corroboration = decision_config.get("strict_corroboration", True)
        self.validation_pipeline = ValidationPipeline(strict_corroboration=strict_corroboration)
        self.decision_engine = DecisionEngine(
            allow_spacy_only_corroboration=not strict_corroboration
        )

        self.classifier = None
        model_path = None
        if config.get("classification", {}).get("use_ml_classifier", True):
            model_path = self.base_dir / config.get("classification", {}).get(
                "model_path", "models/lightgbm/person_classifier.txt"
            )
            self.classifier = load_classifier(model_path)

        # Computed once per process/worker, not per file - see
        # src/utils/provenance.py's module docstring for why this exists.
        self.provenance = build_run_provenance(self.base_dir, model_path)
        spacy_detectors = [d for d in self.detector_manager.detectors if d.name == DetectorName.SPACY]
        self.provenance["spacy_model"] = spacy_detectors[0].model_version if spacy_detectors else None

        self.feedback_enabled = config.get("feedback", {}).get("enabled", True)
        # True in parallel scan workers: build records, return them on the
        # ExtractionResult, and let the main process write them (see
        # FeedbackStore.build_records for why workers must not write).
        self.feedback_defer_writes = config.get("feedback", {}).get("defer_writes", False)
        self.feedback_store = FeedbackStore(self.base_dir / "datasets" / "feedback")

        self.language_filter_enabled = config.get("language_filter", {}).get("enabled", True)

    def extract(
        self,
        cleaned_text: str,
        line_index: LineIndex,
        input_path: str,
        stats: PipelineStatistics,
        source_format: SourceFormat | None = None,
        precomputed_detections: list[Detection] | None = None,
        candidate_filter: Callable[[list[CandidateResult]], list[CandidateResult]] | None = None,
        known_person_names: Iterable[str] = (),
    ) -> ExtractionResult:
        """precomputed_detections / candidate_filter exist for the
        real-time session mode (src/realtime/service.py): a conversation
        window whose older messages were already run through the
        detectors skips re-detecting them (precomputed_detections), still
        builds and boundary-refines candidates across the whole window
        (refinement compares candidates against each other), then only
        validates/classifies/decides the candidates candidate_filter
        keeps - the newest message's, plus earlier mentions whose
        evidence it changed. Defaults reproduce the normal whole-document
        behavior exactly. known_person_names: full names already accepted
        outside this call (a session's earlier messages) - they vouch for
        first names the same way this call's own accepted names do (see
        _propagate_first_names)."""
        candidate_factory = CandidateFactory(line_index, self.knowledge_base)

        # The English-prose gate is calibrated on prose stopword ratios; a
        # structured export (contacts CSV, JSON, XML) is mostly names and
        # values with almost no stopwords and would be skipped wholesale.
        # Per-candidate validators still apply to it as usual.
        if self.language_filter_enabled and source_format not in STRUCTURED_FORMATS:
            with timed_stage("language_filter", stats.stage_timings):
                lang_check = check_english(cleaned_text, self.knowledge_base)
            if not lang_check.is_english:
                logger.info(
                    "Skipping %s: not English-language prose (english_word_ratio=%.4f over "
                    "%d words) - likely non-English legal/UI boilerplate, not evidentiary "
                    "text. Set config['language_filter']['enabled']=False to disable this gate.",
                    input_path, lang_check.english_word_ratio, lang_check.word_count,
                )
                return ExtractionResult(
                    success=True,
                    source_path=input_path,
                    statistics=stats,
                    model_info={
                        "detectors_enabled": [d.name.value for d in self.detector_manager.detectors],
                        "ml_classifier_loaded": self.classifier is not None,
                        "skipped_reason": "non_english",
                        "english_word_ratio": lang_check.english_word_ratio,
                        **self.provenance,
                    },
                )

        with timed_stage("detection", stats.stage_timings):
            if precomputed_detections is not None:
                detections = precomputed_detections
            else:
                detections = self.detector_manager.detect_all(cleaned_text, page_index=0)
            stats.raw_detections = len(detections)

        with timed_stage("candidate_factory", stats.stage_timings):
            all_candidates: list[CandidateResult] = candidate_factory.build(detections)
            stats.candidates_generated = len(all_candidates)

        with timed_stage("boundary_refinement", stats.stage_timings):
            all_candidates = split_double_name_candidates(all_candidates, self.knowledge_base, line_index)
            all_candidates = trim_stopword_edges(all_candidates, self.knowledge_base, line_index)
            all_candidates = trim_dictionary_confirmed_organization_edge(all_candidates, self.knowledge_base, line_index)
            all_candidates = refine_boundaries(all_candidates, line_index)
            # Every candidate span in the text, BEFORE filtering - name
            # propagation must not create a new candidate on text another
            # candidate already covers, whether or not that one is being
            # re-judged in this call (real-time sessions filter most out).
            covered_candidates = list(all_candidates)
            if candidate_filter is not None:
                all_candidates = candidate_filter(all_candidates)

        with timed_stage("validation", stats.stage_timings):
            self.validation_pipeline.reset_document_cache()
            for candidate in all_candidates:
                self.validation_pipeline.run(candidate, self.knowledge_base, cleaned_text)

        with timed_stage("feature_engineering", stats.stage_timings):
            for candidate in all_candidates:
                if candidate.state.is_hard_rejected:
                    continue  # no point computing features for a candidate already rejected
                candidate.state.feature_vector = extract_features(
                    candidate, cleaned_text, self.knowledge_base
                )

        with timed_stage("ml_classification", stats.stage_timings):
            if self.classifier is not None:
                scored = [c for c in all_candidates if c.state.feature_vector is not None]
                results = self.classifier.predict_proba_batch([c.state.feature_vector for c in scored])
                for candidate, classifier_result in zip(scored, results):
                    candidate.state.classifier_result = classifier_result

        with timed_stage("decision_engine", stats.stage_timings):
            for candidate in all_candidates:
                self.decision_engine.decide(candidate, cleaned_text, self.knowledge_base)

        with timed_stage("name_propagation", stats.stage_timings):
            new_candidates = self._propagate_first_names(
                all_candidates, covered_candidates, cleaned_text, line_index, candidate_filter, known_person_names,
            )
            all_candidates.extend(new_candidates)
            stats.candidates_generated += len(new_candidates)

        if source_format in STRUCTURED_FORMATS:
            _cap_lone_single_tokens(all_candidates)

        with timed_stage("aggregation", stats.stage_timings):
            accepted_candidates = [c for c in all_candidates if c.state.decision == Decision.ACCEPTED]
            review_candidates = [c for c in all_candidates if c.state.decision == Decision.REVIEW]
            rejected_candidates = [c for c in all_candidates if c.state.decision == Decision.REJECTED]

            persons = aggregate_by_surface_form(accepted_candidates)
            review_persons = aggregate_by_surface_form(review_candidates)

            low_confidence_candidates = [
                c for c in rejected_candidates
                if (c.state.rejection_reason or "").startswith("corroboration:")
            ]
            low_confidence_persons = aggregate_by_surface_form(low_confidence_candidates)

        deferred_feedback_records: list = []
        if self.feedback_enabled and review_persons and self.feedback_defer_writes:
            with timed_stage("feedback_logging", stats.stage_timings):
                deferred_feedback_records = FeedbackStore.build_records(review_persons, input_path, cleaned_text)
        elif self.feedback_enabled and review_persons:
            with timed_stage("feedback_logging", stats.stage_timings):
                logged_count = self.feedback_store.log_review_persons(review_persons, input_path, cleaned_text)
                if logged_count:
                    logger.info(
                        "%d new candidate(s) logged for review at %s - run "
                        "`python scripts/label_feedback.py` to confirm/correct them.",
                        logged_count, self.feedback_store.pending_path,
                    )

        stats.candidates_accepted = len(accepted_candidates)
        stats.candidates_review = len(review_candidates)
        stats.candidates_rejected = len(rejected_candidates)
        stats.unique_persons_found = len(persons)

        model_info = {
            "detectors_enabled": [d.name.value for d in self.detector_manager.detectors],
            "ml_classifier_loaded": self.classifier is not None,
            "identity_resolution_caveat": IDENTITY_RESOLUTION_CAVEAT,
            **self.provenance,
        }
        if self.classifier is not None and getattr(self.classifier, "training_metrics", None):
            model_info["ml_classifier_validation_accuracy"] = self.classifier.training_metrics.get(
                "validation_accuracy", "n/a"
            )

        logger.info(
            "Pipeline finished for %s: %d persons accepted, %d review, %d rejected "
            "(of %d candidates)",
            input_path, len(persons), len(review_persons), len(rejected_candidates),
            stats.candidates_generated,
        )

        return ExtractionResult(
            success=True,
            source_path=input_path,
            persons=tuple(persons),
            review_persons=tuple(review_persons),
            rejected=tuple(rejected_candidates),
            low_confidence_persons=tuple(low_confidence_persons),
            feedback_records=tuple(deferred_feedback_records),
            statistics=stats,
            model_info=model_info,
        )

    def _propagate_first_names(
        self,
        candidates: list[CandidateResult],
        covered_candidates: list[CandidateResult],
        document_text: str,
        line_index: LineIndex,
        candidate_filter: Callable[[list[CandidateResult]], list[CandidateResult]] | None,
        known_person_names: Iterable[str],
    ) -> list[CandidateResult]:
        """See src/candidate/name_propagation.py. Promotes REVIEW
        candidates in place and returns NEW candidates (occurrences no
        detector found), already validated, classified and decided.
        covered_candidates: every candidate span in the text (pre-filter),
        used only to decide what counts as "already detected"."""
        accepted_full_names = [
            c.candidate.normalized_text for c in candidates if c.state.decision == Decision.ACCEPTED
        ]
        names = propagatable_first_names([*accepted_full_names, *known_person_names], self.knowledge_base)
        if not names:
            return []

        for cr, vouched_by in review_candidates_to_promote(candidates, names):
            detection = document_name_detection(cr.candidate.text, cr.candidate.start, cr.candidate.end, vouched_by)
            cr.candidate = replace(cr.candidate, source_detections=cr.candidate.source_detections + (detection,))
            cr.state.add_evidence(
                "name_propagation", f"First name of '{vouched_by}', accepted in this document",
                detection.confidence,
            )
            self.decision_engine.decide(cr, document_text, self.knowledge_base)

        new_candidates = CandidateFactory(line_index, self.knowledge_base).build(
            undetected_occurrences(document_text, names, covered_candidates)
        )
        if candidate_filter is not None:
            new_candidates = candidate_filter(new_candidates)
        if not new_candidates:
            return []

        for cr in new_candidates:
            self.validation_pipeline.run(cr, self.knowledge_base, document_text)
            if not cr.state.is_hard_rejected:
                cr.state.feature_vector = extract_features(cr, document_text, self.knowledge_base)
        if self.classifier is not None:
            scored = [c for c in new_candidates if c.state.feature_vector is not None]
            for cr, result in zip(scored, self.classifier.predict_proba_batch([c.state.feature_vector for c in scored])):
                cr.state.classifier_result = result
        for cr in new_candidates:
            self.decision_engine.decide(cr, document_text, self.knowledge_base)
        return new_candidates


def _cap_lone_single_tokens(candidates: list[CandidateResult]) -> None:
    """In structured data (CSV/JSON/XML - see STRUCTURED_FORMATS), a
    single-token candidate is moved from ACCEPTED to REVIEW unless a title
    or document-name propagation vouches for it. A lone value in a data
    field has no sentence context, and real Android settings/config files
    are full of values that happen to be list names - measured on a real
    case scan (2026-09-24): "Edit" (a Hungarian first name), "Read",
    "Block", "Handler", "English", "List" were ACCEPTED across dozens of
    preference XML files; 223 of the 263 names only the new formats
    produced were single tokens. Multi-token names are unaffected, and
    REVIEW keeps these visible to a human."""
    for c in candidates:
        if c.state.decision != Decision.ACCEPTED or len(c.candidate.normalized_text.split()) != 1:
            continue
        patterns = {d.metadata.get("pattern") for d in c.candidate.source_detections}
        if patterns & {"titled", DOCUMENT_NAME_PATTERN}:
            continue
        c.state.decision = Decision.REVIEW
        c.state.rejection_reason = (
            "Single word in structured data (CSV/JSON/XML) with no title or full-name support - "
            "a lone field value has no sentence context; held for review"
        )
