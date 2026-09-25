"""
src.classification.base_classifier
=====================================
Interface every ML classifier implements, exactly per the blueprint's
Classification section: train(), predict(), predict_proba(), save(),
load(). Swapping LightGBM for XGBoost or RandomForest later means
writing one new class against this interface - nothing else in the
pipeline changes (Decision Engine talks to BaseClassifier, never to a
concrete model class).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.models import ClassifierResult, FeatureVector


class BaseClassifier(ABC):
    model_name: str = "base"
    model_version: str = "0.0.0"

    @abstractmethod
    def train(self, X: list[list[float]], y: list[int]) -> dict:
        """Train on feature rows X and binary labels y. Returns a dict
        of training metrics (accuracy, precision, recall, f1, ...)."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, feature_vector: FeatureVector) -> ClassifierResult:
        """Return a ClassifierResult with label (0/1) and probability
        for a single candidate's FeatureVector. Never raises - errors
        are captured in ClassifierResult.error."""
        raise NotImplementedError

    def predict_proba_batch(self, feature_vectors: list[FeatureVector]) -> list[ClassifierResult]:
        """Score many candidates at once, results in input order. Default
        just loops predict_proba(); concrete models override this with a
        single vectorized call when their backend supports one. Never
        raises, same contract as predict_proba()."""
        return [self.predict_proba(fv) for fv in feature_vectors]

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        raise NotImplementedError
