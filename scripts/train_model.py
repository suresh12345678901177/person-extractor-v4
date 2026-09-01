"""
scripts/train_model.py
=========================
Trains the LightGBM person-name classifier on the synthetic training
data (run scripts/generate_training_data.py first if
datasets/training/synthetic_training_data.jsonl doesn't exist yet),
saves the model to models/lightgbm/person_classifier.txt, and prints
the real validation metrics achieved.

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.classification.lightgbm_classifier import LightGBMClassifier
from src.features.feature_vector import FEATURE_NAMES


def main() -> None:
    data_path = BASE_DIR / "datasets" / "training" / "synthetic_training_data.jsonl"
    if not data_path.exists():
        print(f"Training data not found at {data_path}")
        print("Run this first: python scripts/generate_training_data.py")
        raise SystemExit(1)

    X: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []

    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            X.append(row["features"])
            y.append(row["label"])
            # Grouped by the feature vector itself, not the source text - see
            # LightGBMClassifier.train's docstring comment: two different
            # names routinely produce an identical numeric feature vector,
            # and the model only ever sees these numbers, never the text.
            groups.append(str(tuple(row["features"])))

    print(f"Loaded {len(X)} training examples ({sum(y)} positive, {len(y) - sum(y)} negative)")
    print(f"Feature order: {FEATURE_NAMES}")

    classifier = LightGBMClassifier()
    metrics = classifier.train(X, y, groups=groups)

    model_path = BASE_DIR / "models" / "lightgbm" / "person_classifier.txt"
    classifier.save(model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {model_path.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()
