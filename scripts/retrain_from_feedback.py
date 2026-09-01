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

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.classification.lightgbm_classifier import LightGBMClassifier
from src.feedback.feedback_store import FeedbackStore

REAL_EXAMPLE_UPWEIGHT = 3


def main() -> None:
    synthetic_path = BASE_DIR / "datasets" / "training" / "synthetic_training_data.jsonl"
    if not synthetic_path.exists():
        print(f"Synthetic training data not found at {synthetic_path}")
        print("Run this first: python scripts/generate_training_data.py")
        raise SystemExit(1)

    store = FeedbackStore(BASE_DIR / "datasets" / "feedback")
    confirmed = store.load_confirmed()

    if not confirmed:
        print("No confirmed feedback yet. Run scripts/label_feedback.py first to")
        print("confirm/correct some REVIEW-bucket candidates from a real run.")
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
    classifier.save(model_path)

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")
    print(f"\nModel updated at: {model_path}")


if __name__ == "__main__":
    main()
