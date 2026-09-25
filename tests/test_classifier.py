"""LightGBMClassifier compatibility across feature-schema growth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.classification.lightgbm_classifier import LightGBMClassifier
from src.core.models import FeatureVector
from src.features.feature_vector import FEATURE_NAMES


def _vector(seed: int) -> FeatureVector:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 3, size=19).tolist()
    return FeatureVector(*values, common_word_ratio=0.5, first_token_is_ambiguous=1, place_or_org_token_count=2)


def test_model_trained_on_older_shorter_schema_reads_leading_columns():
    """Features are only ever appended, so a model saved before the
    latest additions must keep working - scoring on exactly the columns it
    was trained on, ignoring the newer trailing ones."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X_old = rng.integers(0, 3, size=(200, 19)).astype(float)
    y = (X_old[:, 4] > 0).astype(int)
    classifier = LightGBMClassifier()
    classifier._booster = lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5},
                                    lgb.Dataset(X_old, y), num_boost_round=10)

    vectors = [_vector(i) for i in range(5)]
    assert len(vectors[0].as_list()) == len(FEATURE_NAMES) > 19

    expected = classifier._booster.predict(np.array([v.as_list()[:19] for v in vectors]))
    single = [classifier.predict_proba(v) for v in vectors]
    batch = classifier.predict_proba_batch(vectors)
    for s, b, e in zip(single, batch, expected):
        assert s.ok and b.ok
        assert s.probability == b.probability == round(float(e), 4)
