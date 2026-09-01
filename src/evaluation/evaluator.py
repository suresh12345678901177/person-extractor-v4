"""
src.evaluation.evaluator
===========================
Runs the full pipeline against every benchmark file in
`datasets/benchmark/` and computes real, honest precision/recall/F1
against the gold-standard labels - this is what backs the CLI's
`--evaluate` flag and the "Model Accuracy" report section.

Two DIFFERENT "accuracy" numbers exist in this project, and this module
deliberately keeps them from being confused with each other:
  - LightGBMClassifier.training_metrics: the classifier's OWN validation
    accuracy on its synthetic training data (see lightgbm_classifier.py).
  - EvaluationMetrics from THIS module: the end-to-end PIPELINE's
    precision/recall/F1 against real, hand-labeled example text. This is
    the more meaningful number for "does the tool work" and is always
    labeled "end-to-end benchmark" in output to keep it distinct from
    the classifier's internal training metric.

Within THIS module, a third distinction matters just as much (added
2026-08-19 after external review correctly flagged it): ACCEPTED-only
metrics vs. candidate-discovery metrics (ACCEPTED + REVIEW combined).
Every BenchmarkFileResult/BenchmarkReport below carries BOTH, separately,
on purpose - collapsing them into one blended number (as this module
used to do) makes "0.87 recall" sound like automatic-extraction recall
when a large share of it is really "the tool put this in front of a
human for confirmation, not that it auto-extracted it correctly."
Report accepted_metrics when the question is "how good is the automatic
output," and candidate_metrics when the question is "what does the tool
surface for a human to review at all" - never one number for both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.metrics import EvaluationMetrics, compute_span_metrics
from src.pipeline.orchestrator import Pipeline
from src.utils.logger import get_logger

logger = get_logger("evaluation.evaluator")


@dataclass(slots=True)
class BenchmarkFileResult:
    name: str
    accepted_metrics: EvaluationMetrics  # ACCEPTED bucket only - "automatic extraction" quality
    candidate_metrics: EvaluationMetrics  # ACCEPTED + REVIEW - "candidate discovery" quality
    missed_mentions: list[str] = field(default_factory=list)  # missed even counting REVIEW
    false_positive_texts: list[str] = field(default_factory=list)  # wrong among ACCEPTED only


@dataclass(slots=True)
class BenchmarkReport:
    file_results: list[BenchmarkFileResult]
    overall_accepted: EvaluationMetrics
    overall_candidate: EvaluationMetrics


def _find_benchmark_pairs(benchmark_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for txt_path in sorted(benchmark_dir.glob("*.txt")):
        gold_path = txt_path.with_name(txt_path.stem + "_gold.json")
        if gold_path.exists():
            pairs.append((txt_path, gold_path))
    return pairs


def run_benchmark(pipeline: Pipeline, benchmark_dir: str | Path = "datasets/benchmark") -> BenchmarkReport:
    benchmark_dir = Path(benchmark_dir)
    pairs = _find_benchmark_pairs(benchmark_dir)

    if not pairs:
        logger.warning("No benchmark txt/gold pairs found in %s", benchmark_dir)
        empty = compute_span_metrics([], [])
        return BenchmarkReport(file_results=[], overall_accepted=empty, overall_candidate=empty)

    file_results: list[BenchmarkFileResult] = []
    all_accepted: list[tuple[int, int]] = []
    all_candidates: list[tuple[int, int]] = []
    all_gold: list[tuple[int, int]] = []

    for txt_path, gold_path in pairs:
        gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_spans = [(m["start"], m["end"]) for m in gold_data["mentions"]]

        result = pipeline.run(txt_path)
        accepted_spans = [
            (mention.candidate.start, mention.candidate.end)
            for person in result.persons for mention in person.mentions
        ]
        review_spans = [
            (mention.candidate.start, mention.candidate.end)
            for person in result.review_persons for mention in person.mentions
        ]
        candidate_spans = accepted_spans + review_spans

        accepted_metrics = compute_span_metrics(accepted_spans, gold_spans)
        candidate_metrics = compute_span_metrics(candidate_spans, gold_spans)

        missed = [
            m["text"] for m in gold_data["mentions"]
            if not any(not (m["end"] <= ps or m["start"] >= pe) for ps, pe in candidate_spans)
        ]
        fps = [
            f"chars {ps}-{pe}" for ps, pe in accepted_spans
            if not any(not (pe <= gs or ps >= ge) for gs, ge in gold_spans)
        ]

        file_results.append(BenchmarkFileResult(
            name=txt_path.name, accepted_metrics=accepted_metrics, candidate_metrics=candidate_metrics,
            missed_mentions=missed, false_positive_texts=fps,
        ))

        all_accepted.extend(accepted_spans)
        all_candidates.extend(candidate_spans)
        all_gold.extend(gold_spans)

        logger.info(
            "Benchmark %s: accepted precision=%.2f recall=%.2f f1=%.2f | "
            "candidate(+review) precision=%.2f recall=%.2f f1=%.2f",
            txt_path.name,
            accepted_metrics.precision, accepted_metrics.recall, accepted_metrics.f1,
            candidate_metrics.precision, candidate_metrics.recall, candidate_metrics.f1,
        )

    overall_accepted = compute_span_metrics(all_accepted, all_gold)
    overall_candidate = compute_span_metrics(all_candidates, all_gold)
    return BenchmarkReport(file_results=file_results, overall_accepted=overall_accepted, overall_candidate=overall_candidate)
