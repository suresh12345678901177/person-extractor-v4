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

from src.evaluation.metrics import EvaluationMetrics, RecallBreakdown, compute_span_metrics
from src.knowledge.knowledge_base import KnowledgeBase
from src.pipeline.orchestrator import Pipeline
from src.utils.logger import get_logger

logger = get_logger("evaluation.evaluator")

# Which document-type "domain" each benchmark file represents, added
# 2026-09-17 so --evaluate can report accuracy broken out by the kind of
# document actually being processed, not just one blended number - a
# regression concentrated in one domain (as really happened: the
# 2026-09-17 context-feature retrain's entire real-world recall dip
# traced to a single call-log-style file) is invisible in an aggregate
# number and previously took manual per-file digging to even notice.
# A benchmark file not listed here falls into "uncategorized" rather
# than silently being dropped from the breakdown or crashing - a
# deliberate reminder to tag new benchmark files, not a hard requirement.
# Per-file offset range for pooled span metrics (see run_benchmark) - far
# larger than any document, so spans from different files never overlap.
_FILE_SPAN_STRIDE = 10**12

FILE_DOMAINS: dict[str, str] = {
    "benchmark_001.txt": "prose",
    "benchmark_002.txt": "prose",
    "benchmark_003.txt": "chat_or_call_log",
    "benchmark_004.txt": "email",
    "benchmark_real_660.txt": "chat_or_call_log",
    "benchmark_real_675.txt": "chat_or_call_log",
    "benchmark_structured_dump_005.txt": "system_log",
}
# Honest, deliberately-not-hidden gap: no OCR-sourced or PDF-extracted
# benchmark file has hand-labeled gold data yet, so an "ocr" domain
# cannot be reported here without fabricating one - see README's
# per-domain metrics section.


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
    # Per-document-type breakdown (see FILE_DOMAINS above). Keys are
    # domain names; a domain with no benchmark file maps to nothing (not
    # a zero-filled entry), so absence in these dicts means "not
    # represented in the benchmark suite," never "measured at zero."
    domain_accepted: dict[str, EvaluationMetrics] = field(default_factory=dict)
    domain_candidate: dict[str, EvaluationMetrics] = field(default_factory=dict)
    # Per-shape recall breakdown, pooled across every benchmark file -
    # see RecallBreakdown's docstring for why this is recall-only.
    shape_recall_accepted: dict[str, RecallBreakdown] = field(default_factory=dict)
    shape_recall_candidate: dict[str, RecallBreakdown] = field(default_factory=dict)


def _gold_mention_shapes(text: str, kb: KnowledgeBase) -> tuple[str, str]:
    """Returns (token-count category, dictionary category) for one gold
    mention's text - the two shape dimensions --evaluate breaks recall
    out by. A mention contributes to one bucket in EACH dimension (e.g.
    a single-token unseen name counts once under "single_token" and once
    under "unseen_name", not as a combined 4th bucket - keeps each
    dimension's total meaningful on its own rather than fragmenting into
    many small combined buckets)."""
    tokens = text.split()
    token_shape = "single_token" if len(tokens) == 1 else "multi_token"
    is_known = any(kb.is_known_first_name(t) or kb.is_known_last_name(t) for t in tokens)
    dict_shape = "dictionary_known" if is_known else "unseen_name"
    return token_shape, dict_shape


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

    knowledge_base = pipeline.extractor.knowledge_base

    file_results: list[BenchmarkFileResult] = []
    all_accepted: list[tuple[int, int]] = []
    all_candidates: list[tuple[int, int]] = []
    all_gold: list[tuple[int, int]] = []

    domain_accepted_spans: dict[str, list[tuple[int, int]]] = {}
    domain_candidate_spans: dict[str, list[tuple[int, int]]] = {}
    domain_gold_spans: dict[str, list[tuple[int, int]]] = {}

    shape_accepted_hits: dict[str, int] = {}
    shape_accepted_total: dict[str, int] = {}
    shape_candidate_hits: dict[str, int] = {}
    shape_candidate_total: dict[str, int] = {}

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

        # Pooled (overall / per-domain) metrics need spans from different
        # files kept apart: spans are bare char offsets, so without this a
        # prediction at chars 316-324 in one file could "match" a gold
        # mention at the same offsets in ANOTHER file, and
        # compute_span_metrics' greedy one-to-one matching let such cross-
        # file pairs consume each other (found 2026-09-24: overall recall
        # moved while every per-file row stayed identical). Shifting each
        # file into its own offset range makes the pooled numbers exactly
        # the sum of the per-file matches.
        base = len(file_results) * _FILE_SPAN_STRIDE
        accepted_pooled = [(s + base, e + base) for s, e in accepted_spans]
        candidate_pooled = [(s + base, e + base) for s, e in candidate_spans]
        gold_pooled = [(s + base, e + base) for s, e in gold_spans]

        all_accepted.extend(accepted_pooled)
        all_candidates.extend(candidate_pooled)
        all_gold.extend(gold_pooled)

        domain = FILE_DOMAINS.get(txt_path.name, "uncategorized")
        domain_accepted_spans.setdefault(domain, []).extend(accepted_pooled)
        domain_candidate_spans.setdefault(domain, []).extend(candidate_pooled)
        domain_gold_spans.setdefault(domain, []).extend(gold_pooled)

        for mention in gold_data["mentions"]:
            ms, me = mention["start"], mention["end"]
            found_accepted = any(not (me <= ps or ms >= pe) for ps, pe in accepted_spans)
            found_candidate = any(not (me <= ps or ms >= pe) for ps, pe in candidate_spans)
            for shape in _gold_mention_shapes(mention["text"], knowledge_base):
                shape_accepted_total[shape] = shape_accepted_total.get(shape, 0) + 1
                shape_candidate_total[shape] = shape_candidate_total.get(shape, 0) + 1
                if found_accepted:
                    shape_accepted_hits[shape] = shape_accepted_hits.get(shape, 0) + 1
                if found_candidate:
                    shape_candidate_hits[shape] = shape_candidate_hits.get(shape, 0) + 1

        logger.info(
            "Benchmark %s: accepted precision=%.2f recall=%.2f f1=%.2f | "
            "candidate(+review) precision=%.2f recall=%.2f f1=%.2f",
            txt_path.name,
            accepted_metrics.precision, accepted_metrics.recall, accepted_metrics.f1,
            candidate_metrics.precision, candidate_metrics.recall, candidate_metrics.f1,
        )

    overall_accepted = compute_span_metrics(all_accepted, all_gold)
    overall_candidate = compute_span_metrics(all_candidates, all_gold)

    domain_accepted = {
        domain: compute_span_metrics(domain_accepted_spans[domain], domain_gold_spans[domain])
        for domain in domain_gold_spans
    }
    domain_candidate = {
        domain: compute_span_metrics(domain_candidate_spans[domain], domain_gold_spans[domain])
        for domain in domain_gold_spans
    }
    shape_recall_accepted = {
        shape: RecallBreakdown(hits=shape_accepted_hits.get(shape, 0), total=total)
        for shape, total in shape_accepted_total.items()
    }
    shape_recall_candidate = {
        shape: RecallBreakdown(hits=shape_candidate_hits.get(shape, 0), total=total)
        for shape, total in shape_candidate_total.items()
    }

    return BenchmarkReport(
        file_results=file_results, overall_accepted=overall_accepted, overall_candidate=overall_candidate,
        domain_accepted=domain_accepted, domain_candidate=domain_candidate,
        shape_recall_accepted=shape_recall_accepted, shape_recall_candidate=shape_recall_candidate,
    )
