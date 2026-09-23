"""
src.evaluation.metrics
=========================
Span-level precision/recall/F1 computation: compares the pipeline's
accepted candidate spans against a labeled gold-standard set of
(char_start, char_end) person-mention spans, using overlap (not exact
match) as the hit criterion - a predicted span that overlaps a gold
span at all counts as a match, since detectors may include/exclude a
title prefix differently than the label without that being a genuine
error worth penalizing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    gold_total: int
    predicted_total: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return round(self.true_positives / denom, 4) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return round(self.true_positives / denom, 4) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        """Span-level accuracy: TP / (TP + FP + FN). There is no
        meaningful "true negative" at the span level (the space of
        non-person spans in free text is unbounded), so accuracy here
        is reported on this widely-used NER convention, not the
        classic (TP+TN)/(all) formula - documented explicitly so the
        number is never misread as classifier-style accuracy."""
        denom = self.true_positives + self.false_positives + self.false_negatives
        return round(self.true_positives / denom, 4) if denom else 0.0

    def as_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "gold_total": self.gold_total,
            "predicted_total": self.predicted_total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


@dataclass(slots=True)
class RecallBreakdown:
    """Recall-only metric for a gold-mention SUBSET (e.g. "single-token
    names", "names not in any dictionary"). Precision deliberately isn't
    tracked here: a false positive has no gold mention to categorize
    against, so "precision for the single-token subset" isn't a
    well-defined question the way "recall for the single-token subset"
    is (of the gold mentions that are single-token, how many did we
    find) - see evaluator.py's per-shape breakdown for where this is
    used."""
    hits: int
    total: int

    @property
    def recall(self) -> float:
        return round(self.hits / self.total, 4) if self.total else 0.0

    def as_dict(self) -> dict:
        return {"hits": self.hits, "total": self.total, "recall": self.recall}


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def compute_span_metrics(
    predicted_spans: list[tuple[int, int]], gold_spans: list[tuple[int, int]]
) -> EvaluationMetrics:
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for gi, (gs, ge) in enumerate(gold_spans):
        for pi, (ps, pe) in enumerate(predicted_spans):
            if pi in matched_pred:
                continue
            if _overlaps(gs, ge, ps, pe):
                matched_gold.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gold)
    fn = len(gold_spans) - tp
    fp = len(predicted_spans) - len(matched_pred)

    return EvaluationMetrics(
        true_positives=tp, false_positives=fp, false_negatives=fn,
        gold_total=len(gold_spans), predicted_total=len(predicted_spans),
    )
