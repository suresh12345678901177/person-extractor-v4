"""
scripts/retrain_from_feedback.py
===================================
Retrains the LightGBM classifier using your REAL human-confirmed
corrections (from scripts/label_feedback.py) combined with the original
synthetic distant-supervision data. This is what closes the active
learning loop: instead of the classifier being permanently frozen at
whatever the synthetic bootstrap taught it, it incorporates genuine
examples from documents you've actually run.

Real confirmed examples are duplicated 3x in the training set relative
to synthetic ones - a deliberate, documented choice: with only a
handful of real corrections against thousands of synthetic examples,
the real signal would otherwise be drowned out. This is a standard,
simple way to upweight scarce high-quality labels without needing a
more complex weighted-loss training setup.

Usage:
    python scripts/retrain_from_feedback.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path, PurePosixPath

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG
from src.classification.lightgbm_classifier import LightGBMClassifier
from src.evaluation.evaluator import run_benchmark
from src.feedback.feedback_store import FeedbackRecord, FeedbackStore
from src.pipeline.orchestrator import Pipeline

REAL_EXAMPLE_UPWEIGHT = 3

# Regression guardrail (added 2026-09-22, after this exact failure mode
# happened TWICE in this project's real history: 2026-09-16 (133 real
# labels, synthetic validation accuracy improved 0.9324->0.9737 but real
# accepted-only recall on --evaluate REGRESSED 0.3062->0.1849) and again
# 2026-09-22 (a 27-negative/6-positive batch measurably dropped
# dictionary_known accepted-only recall 906/930->847/930, caught only by
# manually running --evaluate after the fact and reverting from backup).
# `save()` already backs up the file it's about to overwrite - this adds
# the other half: evaluate the CANDIDATE model against the real
# --evaluate benchmark BEFORE it ever touches the live model path, and
# refuse to auto-promote it if any candidate-shape accepted-only recall
# drops by more than this many percentage points with no offsetting
# overall improvement.
MAX_ACCEPTABLE_SHAPE_RECALL_DROP = 0.02

# Every file --evaluate and scripts/evaluate_unseen_names.py score against,
# matched by folder OR file name (2026-09-25): the folder-only check missed
# datasets/benchmark_unseen/ - the held-out set, one of whose names was
# already sitting in pending_review.jsonl - and any copy of a benchmark file
# run from another folder (the by-file-name rule auto_label_from_gold.py
# already uses).
_EVALUATION_DIRS = ("benchmark", "benchmark_unseen")
_EVALUATION_FILE_NAMES = frozenset(
    p.name.lower() for d in _EVALUATION_DIRS for p in (BASE_DIR / "datasets" / d).glob("*.txt")
)


def _is_benchmark_source(source_file: str) -> bool:
    """True if this feedback record was logged from a file under
    datasets/benchmark/. Those exact files are what `python cli.py
    --evaluate` scores the pipeline against, so training on them - even
    more so 3x-upweighted - leaks the evaluation domain into training
    and inflates/distorts the reported benchmark numbers for those
    specific files. Found 2026-09-17: 229 of 1,276 confirmed labels
    (124 from benchmark_real_660.txt, 66 from benchmark_real_675.txt,
    39 from the four small hand-authored benchmark_0*.txt files) came
    from exactly this leak. Excluded here so --evaluate measures
    genuine generalization, not partial memorization."""
    # PurePosixPath on "/"-normalized text: records store Windows paths,
    # which a POSIX Path (e.g. in CI) would not split on "\".
    path = PurePosixPath(source_file.replace("\\", "/"))
    return path.parent.name.lower() in _EVALUATION_DIRS or path.name.lower() in _EVALUATION_FILE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the classifier from real feedback")
    parser.add_argument("--force", action="store_true",
                         help="Promote the retrained model even if the regression guardrail "
                              "finds a shape-recall drop with no offsetting overall improvement")
    args = parser.parse_args()

    synthetic_path = BASE_DIR / "datasets" / "training" / "synthetic_training_data.jsonl"
    if not synthetic_path.exists():
        print(f"Synthetic training data not found at {synthetic_path}")
        print("Run this first: python scripts/generate_training_data.py")
        raise SystemExit(1)

    store = FeedbackStore(BASE_DIR / "datasets" / "feedback")
    all_confirmed = store.load_confirmed()

    if not all_confirmed:
        print("No confirmed feedback yet. Run scripts/label_feedback.py first to")
        print("confirm/correct some REVIEW-bucket candidates from a real run.")
        raise SystemExit(1)

    confirmed: list[FeedbackRecord] = []
    excluded_benchmark: list[FeedbackRecord] = []
    for record in all_confirmed:
        if _is_benchmark_source(record.source_file):
            excluded_benchmark.append(record)
        else:
            confirmed.append(record)

    if excluded_benchmark:
        print(f"Excluding {len(excluded_benchmark)} confirmed label(s) sourced from "
              f"benchmark files (datasets/benchmark/, datasets/benchmark_unseen/ - see "
              f"_is_benchmark_source) - these files are what the evaluations score against.")
    if not confirmed:
        print("No non-benchmark confirmed feedback left after exclusion.")
        raise SystemExit(1)

    X: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []

    with synthetic_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            X.append(row["features"])
            y.append(row["label"])
            # Grouped by feature vector, not source text - see
            # LightGBMClassifier.train's docstring: different names routinely
            # collide on this coarse a feature representation, and the model
            # only ever sees these numbers, never the text.
            groups.append(str(tuple(row["features"])))

    synthetic_count = len(X)

    real_positive = 0
    real_negative = 0
    for record in confirmed:
        label = 1 if record.status == "confirmed_person" else 0
        # Same group key for all 3 upweighted copies (and for any synthetic
        # row that happens to share this exact feature vector) - a
        # group-aware split (see LightGBMClassifier.train) then guarantees
        # they all land on the SAME side, never split across train/val.
        for _ in range(REAL_EXAMPLE_UPWEIGHT):
            X.append(record.features)
            y.append(label)
            groups.append(str(tuple(record.features)))
        if label == 1:
            real_positive += 1
        else:
            real_negative += 1

    print(f"Synthetic examples:        {synthetic_count}")
    print(f"Real confirmed examples:   {len(confirmed)} "
          f"({real_positive} person, {real_negative} not-person) "
          f"- upweighted {REAL_EXAMPLE_UPWEIGHT}x -> {len(confirmed) * REAL_EXAMPLE_UPWEIGHT} training rows")
    print(f"Total training examples:   {len(X)}")

    classifier = LightGBMClassifier()
    metrics = classifier.train(X, y, groups=groups)
    metrics["real_confirmed_examples"] = len(confirmed)
    metrics["real_positive"] = real_positive
    metrics["real_negative"] = real_negative
    metrics["synthetic_examples"] = synthetic_count
    classifier.training_metrics = metrics

    model_path = BASE_DIR / "models" / "lightgbm" / "person_classifier.txt"
    candidate_path = BASE_DIR / "models" / "lightgbm" / "person_classifier_candidate.txt"

    # Regression guardrail (see MAX_ACCEPTABLE_SHAPE_RECALL_DROP's
    # module-level docstring for why this exists): save to a SCRATCH
    # path first and evaluate it against the real --evaluate benchmark
    # before it ever touches the live model path, so a regression is
    # caught automatically instead of requiring someone to remember to
    # run --evaluate by hand afterward.
    classifier.save(candidate_path)  # scratch path - save()'s own backup-on-overwrite is a no-op here

    candidate_config = copy.deepcopy(DEFAULT_CONFIG)
    candidate_config["classification"]["model_path"] = "models/lightgbm/person_classifier_candidate.txt"
    candidate_pipeline = Pipeline(config=candidate_config, base_dir=BASE_DIR)

    print("\nEvaluating candidate model against datasets/benchmark/ before promoting it...")
    candidate_report = run_benchmark(candidate_pipeline)

    promote = True
    if model_path.exists():
        baseline_pipeline = Pipeline(config=copy.deepcopy(DEFAULT_CONFIG), base_dir=BASE_DIR)
        baseline_report = run_benchmark(baseline_pipeline)

        print("\nPer-shape accepted-only recall, live model -> candidate model:")
        regressions = []
        all_shapes = sorted(set(baseline_report.shape_recall_accepted) | set(candidate_report.shape_recall_accepted))
        for shape in all_shapes:
            before = baseline_report.shape_recall_accepted.get(shape)
            after = candidate_report.shape_recall_accepted.get(shape)
            before_r = before.recall if before else 0.0
            after_r = after.recall if after else 0.0
            delta = after_r - before_r
            flag = ""
            if delta < -MAX_ACCEPTABLE_SHAPE_RECALL_DROP:
                flag = "  <-- REGRESSION"
                regressions.append((shape, before_r, after_r, delta))
            print(f"  {shape:<18} {before_r:.4f} -> {after_r:.4f}  (delta {delta:+.4f}){flag}")

        overall_before = baseline_report.overall_accepted.f1
        overall_after = candidate_report.overall_accepted.f1
        print(f"\nOverall accepted-only F1: {overall_before:.4f} -> {overall_after:.4f} "
              f"(delta {overall_after - overall_before:+.4f})")

        overall_improved = overall_after >= overall_before
        if regressions and not overall_improved:
            promote = False
            print("\n" + "!" * 70)
            print("REFUSING TO PROMOTE: shape-recall regression(s) above with no offsetting")
            print("overall F1 improvement.")
            print(f"Candidate model saved for inspection at: {candidate_path}")
            print(f"LIVE model at {model_path} was NOT touched.")
            if args.force:
                print("--force passed: promoting anyway.")
                promote = True
            else:
                print("Re-run with --force to promote it anyway.")
            print("!" * 70)
    else:
        print("\nNo existing live model to compare against - promoting unconditionally.")

    if not promote:
        return

    classifier.save(model_path)  # backs up the previous live model automatically
    candidate_path.unlink(missing_ok=True)
    candidate_metrics_path = candidate_path.with_suffix(".metrics.json")
    candidate_metrics_path.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")
    print(f"\nModel updated at: {model_path}")


if __name__ == "__main__":
    main()
