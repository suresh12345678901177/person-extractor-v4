"""
src.classification.lightgbm_classifier
=========================================
LightGBM binary classifier: "is this candidate span a real person
name?" Chosen over XGBoost/RandomForest for V4 because it trains fast
on the modest feature-vector dataset this project produces, handles the
mix of continuous (confidence scores, ratios) and count features here
without preprocessing, and ships as a small, dependency-light artifact
that is trivial to bundle offline (no CUDA, no large runtime).

IMPORTANT - what this model actually is: it is trained via distant
supervision on a SYNTHETICALLY GENERATED dataset (see
scripts/generate_training_data.py), not a hand-labeled gold corpus.
Positive examples are built from the knowledge base's own name lists;
negative examples are built from the same organization/location/
campaign/blacklist lists used by the rules engine, plus generic
capitalized-bigram distractors. This is a legitimate and common
technique (weak/distant supervision) for bootstrapping a model when no
labeled corpus exists, but it means the model has learned to agree with
the rules engine's own definitions, not to generalize beyond them - it
is used ONLY as a secondary corroborating signal in the Decision
Engine, never as a hard gate, and this scope is documented in
README.md and reported honestly in the CLI's "Model Accuracy" section.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.models import ClassifierResult, FeatureVector
from src.classification.base_classifier import BaseClassifier
from src.utils.logger import get_logger

logger = get_logger("classification.lightgbm_classifier")


class LightGBMClassifier(BaseClassifier):
    model_name = "lightgbm_person_classifier"
    model_version = "1.0.0"

    def __init__(self) -> None:
        self._booster = None
        self.training_metrics: dict = {}

    def train(self, X: list[list[float]], y: list[int], groups: list[str] | None = None) -> dict:
        import numpy as np
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        if groups is not None:
            # GROUP-aware split (2026-08-19, after external review found real
            # validation leakage): every row sharing the same group goes
            # entirely to ONE side of the split. Plain train_test_split split
            # individual ROWS at random, which let exact duplicates -
            # especially the 3x-upweighted real feedback rows in
            # retrain_from_feedback.py - land on both sides, so validation
            # accuracy was partly the model "recognizing" a row it was also
            # trained on (verified: 287/495 validation rows had the identical
            # text+label present in training under the old split).
            #
            # The group key passed in is the FEATURE VECTOR itself, not the
            # source text - verified this matters: grouping by text alone
            # closed the exact-duplicate-row leak but left a second, deeper
            # one open. This feature representation is coarse (16 mostly
            # small-integer/ratio/boolean values), so two DIFFERENT real
            # names routinely produce an IDENTICAL numeric feature vector -
            # measured directly after switching to text-grouping: 305 of 479
            # validation rows still had their exact feature-vector+label
            # combination sitting in training under a different name. The
            # model only ever sees these 16 numbers, never the text, so
            # grouping by feature vector is what actually guarantees no
            # validation ROW's input pattern was seen during training.
            group_arr = np.array(groups)
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(splitter.split(X_arr, y_arr, groups=group_arr))
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
            )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "learning_rate": 0.08,
            "min_data_in_leaf": 5,
            "feature_pre_filter": False,
        }

        self._booster = lgb.train(
            params,
            train_data,
            num_boost_round=150,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        val_probs = self._booster.predict(X_val)
        val_preds = [1 if p >= 0.5 else 0 for p in val_probs]

        metrics = {
            "validation_accuracy": round(accuracy_score(y_val, val_preds), 4),
            "validation_precision": round(precision_score(y_val, val_preds, zero_division=0), 4),
            "validation_recall": round(recall_score(y_val, val_preds, zero_division=0), 4),
            "validation_f1": round(f1_score(y_val, val_preds, zero_division=0), 4),
            "train_examples": len(X_train),
            "validation_examples": len(X_val),
            "positive_examples_total": int(sum(y)),
            "negative_examples_total": int(len(y) - sum(y)),
            "best_iteration": self._booster.best_iteration,
        }
        self.training_metrics = metrics
        logger.info("LightGBM training complete: %s", metrics)
        return metrics

    def predict_proba(self, feature_vector: FeatureVector) -> ClassifierResult:
        if self._booster is None:
            return ClassifierResult(
                label=0, probability=0.0, model_name=self.model_name,
                model_version=self.model_version, error="Model not loaded/trained",
            )
        try:
            import numpy as np
            row = np.array([feature_vector.as_list()], dtype=float)
            prob = float(self._booster.predict(row)[0])
            label = 1 if prob >= 0.5 else 0
            return ClassifierResult(
                label=label, probability=round(prob, 4),
                model_name=self.model_name, model_version=self.model_version,
            )
        except Exception as exc:  # noqa: BLE001 - classifier must never crash the pipeline
            logger.exception("LightGBM prediction failed")
            return ClassifierResult(
                label=0, probability=0.0, model_name=self.model_name,
                model_version=self.model_version, error=str(exc),
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))

        metrics_path = path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps({
            "model_name": self.model_name,
            "model_version": self.model_version,
            **self.training_metrics,
        }, indent=2), encoding="utf-8")
        logger.info("Model saved to %s (metrics: %s)", path, metrics_path)

    def load(self, path: str | Path) -> None:
        import lightgbm as lgb

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._booster = lgb.Booster(model_file=str(path))

        metrics_path = path.with_suffix(".metrics.json")
        if metrics_path.exists():
            self.training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        logger.info("Model loaded from %s", path)
