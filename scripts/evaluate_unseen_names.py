"""
scripts/evaluate_unseen_names.py
===================================
Runs the full pipeline against datasets/benchmark_unseen/ ONLY - a
standalone benchmark where every person name is verified absent from
every assets/*.txt list (see build_unseen_gold_labels.py) - and reports
real precision/recall/F1, same computation as cli.py --evaluate but
scoped to this one directory so it never mixes with, or silently
changes, the official datasets/benchmark/ numbers documented in
README.md.

Usage:
    python scripts/evaluate_unseen_names.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG
from src.evaluation.evaluator import run_benchmark
from src.pipeline.orchestrator import Pipeline
from src.utils.logger import configure_logging


def main() -> int:
    configure_logging(BASE_DIR / "logs")
    # No feedback logging: this is the held-out set, and every REVIEW name it
    # logged became a labeling candidate - a direct path for held-out names
    # into training data.
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    pipeline = Pipeline(config=config, base_dir=BASE_DIR)

    bench_dir = BASE_DIR / "datasets" / "benchmark_unseen"
    print("=" * 78)
    print("UNSEEN-NAMES BENCHMARK - accuracy WITHOUT dictionary support")
    print("=" * 78)
    print(f"Directory: {bench_dir}")
    print("Every person name here is verified absent from every assets/*.txt")
    print("list - this isolates what regex shape + spaCy + repetition/title")
    print("corroboration can do on their own, with zero dictionary crutch.")

    report = run_benchmark(pipeline, bench_dir)

    if not report.file_results:
        print("No benchmark files found - run scripts/build_unseen_gold_labels.py first.")
        return 1

    for fr in report.file_results:
        print()
        print("-" * 78)
        print(f"FILE: {fr.name}")
        print("-" * 78)
        a, c = fr.accepted_metrics, fr.candidate_metrics
        print(f"  Gold mentions               : {a.gold_total}")
        print(f"  -- ACCEPTED only (automatic extraction quality) --")
        print(f"    Predicted / TP / FP / FN  : {a.predicted_total} / {a.true_positives} / {a.false_positives} / {a.false_negatives}")
        print(f"    Precision / Recall / F1   : {a.precision:.4f} / {a.recall:.4f} / {a.f1:.4f}")
        print(f"  -- ACCEPTED + REVIEW (candidate-discovery quality) --")
        print(f"    Predicted / TP / FP / FN  : {c.predicted_total} / {c.true_positives} / {c.false_positives} / {c.false_negatives}")
        print(f"    Precision / Recall / F1   : {c.precision:.4f} / {c.recall:.4f} / {c.f1:.4f}")
        if fr.missed_mentions:
            print(f"  Missed even w/ review        : {fr.missed_mentions}")
        if fr.false_positive_texts:
            print(f"  False positives (accepted)  : {fr.false_positive_texts}")

    print()
    print("=" * 78)
    print("OVERALL (this directory only - NOT the official datasets/benchmark suite)")
    print("=" * 78)
    a, c = report.overall_accepted, report.overall_candidate
    print(f"  ACCEPTED only:")
    print(f"    Precision : {a.precision:.4f}")
    print(f"    Recall    : {a.recall:.4f}")
    print(f"    F1        : {a.f1:.4f}")
    print(f"  ACCEPTED + REVIEW:")
    print(f"    Precision : {c.precision:.4f}")
    print(f"    Recall    : {c.recall:.4f}")
    print(f"    F1        : {c.f1:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
