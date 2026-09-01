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

from pathlib import Path
from typing import Any

from src.candidate.boundary_refiner import (
    refine_boundaries,
    split_double_name_candidates,
    trim_dictionary_confirmed_organization_edge,
    trim_stopword_edges,
)
from src.candidate.candidate_aggregator import aggregate_by_surface_form
from src.candidate.candidate_factory import CandidateFactory
from src.classification.model_registry import load_classifier
from src.core.models import CandidateResult, Decision, ExtractionResult, PipelineStatistics
from src.decision.decision_engine import DecisionEngine
from src.detection.detector_manager import DetectorManager
from src.extraction.base_extractor import Extractor
from src.features.feature_extractor import extract_features
from src.feedback.feedback_store import FeedbackStore
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.line_indexer import LineIndex
from src.utils.logger import get_logger
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
        if config.get("classification", {}).get("use_ml_classifier", True):
            model_path = self.base_dir / config.get("classification", {}).get(
                "model_path", "models/lightgbm/person_classifier.txt"
            )
            self.classifier = load_classifier(model_path)

        self.feedback_enabled = config.get("feedback", {}).get("enabled", True)
        self.feedback_store = FeedbackStore(self.base_dir / "datasets" / "feedback")

    def extract(
        self,
        cleaned_text: str,
        line_index: LineIndex,
        input_path: str,
        stats: PipelineStatistics,
    ) -> ExtractionResult:
        candidate_factory = CandidateFactory(line_index)

        with timed_stage("detection", stats.stage_timings):
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

        with timed_stage("validation", stats.stage_timings):
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
                for candidate in all_candidates:
                    if candidate.state.feature_vector is not None:
                        candidate.state.classifier_result = self.classifier.predict_proba(
                            candidate.state.feature_vector
                        )

        with timed_stage("decision_engine", stats.stage_timings):
            for candidate in all_candidates:
                self.decision_engine.decide(candidate, cleaned_text)

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

        if self.feedback_enabled and review_persons:
            with timed_stage("feedback_logging", stats.stage_timings):
                logged_count = self.feedback_store.log_review_persons(review_persons, input_path)
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
            statistics=stats,
            model_info=model_info,
        )
