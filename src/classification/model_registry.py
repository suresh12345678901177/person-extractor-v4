"""
src.classification.model_registry
====================================
Single place the pipeline asks for "the current classifier". Loads the
saved LightGBM model from disk once and hands back a ready-to-use
instance. If no trained model file exists yet (e.g. a fresh checkout
before `python scripts/train_model.py` has been run), returns None and
logs a clear warning rather than crashing - the pipeline degrades
gracefully to rules-only scoring in that case (see DecisionEngine).
"""

from __future__ import annotations

from pathlib import Path

from src.classification.base_classifier import BaseClassifier
from src.classification.lightgbm_classifier import LightGBMClassifier
from src.utils.logger import get_logger

logger = get_logger("classification.model_registry")

DEFAULT_MODEL_PATH = Path("models/lightgbm/person_classifier.txt")


def load_classifier(model_path: str | Path = DEFAULT_MODEL_PATH) -> BaseClassifier | None:
    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning(
            "No trained model found at %s - run `python scripts/train_model.py` "
            "to train one. Falling back to rules-only scoring.", model_path
        )
        return None

    classifier = LightGBMClassifier()
    try:
        classifier.load(model_path)
    except Exception:
        logger.exception("Failed to load classifier from %s - falling back to rules-only scoring.", model_path)
        return None
    return classifier
