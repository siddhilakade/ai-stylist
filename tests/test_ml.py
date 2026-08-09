"""Tests for the ML compatibility experiment.

Two jobs:
  1. Verify the model loads, predicts sanely, and uses exactly the features it
     was trained on.
  2. Verify - critically - that the model is NOT affecting recommendations.
     The experiment did not justify integration, so "the ML is not in the
     ranking path" is a property that must be enforced by a test, not by memory.
"""

from __future__ import annotations

import pytest

from src.data import catalog_records
from src.ml import model as ml_model
from src.ml.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    canonical_order,
    pair_features,
)

pytestmark = pytest.mark.skipif(
    not ml_model.model_exists(),
    reason="model artifact absent; run scripts/train_compatibility_model.py",
)


@pytest.fixture(scope="module")
def sample_pair():
    products = list(catalog_records())
    top = next(p for p in products if p["outfit_slot"] == "top")
    bottom = next(p for p in products if p["outfit_slot"] == "bottom")
    return top, bottom


class TestModelLoading:
    def test_model_loads(self):
        bundle = ml_model.load_model()
        assert "pipeline" in bundle
        assert bundle["feature_columns"] == FEATURE_COLUMNS

    def test_load_is_cached(self):
        assert ml_model.load_model() is ml_model.load_model()

    def test_bundle_records_that_labels_are_synthetic(self):
        # The artifact must carry its own provenance, so a model file found on
        # its own can never be mistaken for one trained on real behaviour.
        assert "synthetic" in ml_model.load_model()["label_provenance"].lower()

    def test_artifact_is_small_enough_to_commit(self):
        assert ml_model.MODEL_PATH.stat().st_size < 20_000_000


class TestPrediction:
    def test_returns_a_probability(self, sample_pair):
        probability = ml_model.predict_compatibility(*sample_pair)
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

    # The forest runs with n_jobs=-1, so per-tree votes are summed in whatever
    # order the threads finish. That moves the result by ~1e-16 - float
    # associativity, not nondeterministic behaviour - so these compare with a
    # tolerance rather than by exact equality.
    TOLERANCE = 1e-12

    def test_is_deterministic(self, sample_pair):
        first = ml_model.predict_compatibility(*sample_pair)
        second = ml_model.predict_compatibility(*sample_pair)
        assert first == pytest.approx(second, abs=self.TOLERANCE)

    def test_is_symmetric(self, sample_pair):
        """Compatibility has no direction; canonical ordering must guarantee it."""
        a, b = sample_pair
        assert ml_model.predict_compatibility(a, b) == pytest.approx(
            ml_model.predict_compatibility(b, a), abs=self.TOLERANCE
        )

    def test_batch_matches_single(self, sample_pair):
        a, b = sample_pair
        batch = ml_model.predict_many([(a, b), (b, a)])
        assert batch[0] == pytest.approx(ml_model.predict_compatibility(a, b))
        assert batch[0] == pytest.approx(batch[1])

    def test_empty_batch(self):
        assert ml_model.predict_many([]) == []

    def test_separates_an_obvious_clash_from_an_obvious_match(self):
        """A directional sanity check, not an accuracy claim."""
        products = list(catalog_records())
        formal_shoes = next(
            (p for p in products if p["articleType"] == "Formal Shoes"), None
        )
        shorts = next((p for p in products if p["articleType"] == "Shorts"), None)
        formal_trousers = next(
            (p for p in products
             if p["articleType"] == "Trousers" and p["usage"] == "Formal"), None
        )
        if not all((formal_shoes, shorts, formal_trousers)):
            pytest.skip("catalog lacks the items for this check")

        clash = ml_model.predict_compatibility(formal_shoes, shorts)
        match = ml_model.predict_compatibility(formal_shoes, formal_trousers)
        assert match > clash

    def test_safe_predict_never_raises(self, sample_pair):
        assert ml_model.safe_predict(*sample_pair) is not None
        assert ml_model.safe_predict({}, {}) is None


class TestFeatureConsistency:
    def test_feature_columns_are_the_union_of_the_three_groups(self):
        assert set(FEATURE_COLUMNS) == (
            set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) | set(BOOLEAN_FEATURES)
        )
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_inference_features_match_training_features(self, sample_pair):
        assert set(pair_features(*sample_pair)) == set(FEATURE_COLUMNS)

    def test_no_rule_derived_feature_leaks_in(self):
        """The anti-leakage contract, enforced.

        If a rule score ever appears as a feature, the model stops being an
        independent estimator and the whole experiment becomes circular.
        """
        forbidden = (
            "rule_score", "compatibility", "color_score", "colour_score",
            "formality_score", "occasion_score", "ethnic_score",
            "compatible", "pair_compatibility", "formality_gap",
        )
        for feature in FEATURE_COLUMNS:
            assert feature not in forbidden, feature

    def test_canonical_order_is_stable(self, sample_pair):
        a, b = sample_pair
        assert canonical_order(a, b) == canonical_order(b, a)


class TestNotInProductionPath:
    """The experiment did not justify shipping. Prove it stays unshipped."""

    def test_module_declares_itself_out_of_the_ranking_path(self):
        assert ml_model.IS_IN_PRODUCTION_PATH is False

    def test_compatibility_module_does_not_import_the_model(self):
        import src.compatibility as compatibility

        source = compatibility.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "ml.model" not in text and "predict_compatibility" not in text

    def test_recommender_does_not_import_the_model(self):
        import src.recommender as recommender

        with open(recommender.__file__, encoding="utf-8") as handle:
            text = handle.read()
        assert "ml.model" not in text and "predict_compatibility" not in text

    def test_recommendations_are_unchanged_by_the_models_presence(self):
        """Recommendations must be identical whether or not the model loads."""
        from src.recommender import recommend_outfits
        from src.schemas import StylePreferences

        prefs = StylePreferences(gender="Men", occasion="work_office",
                                 style="formal", budget=8000)
        before = [o.product_ids for o in recommend_outfits(prefs).outfits]

        # Force the model into memory, then recommend again.
        ml_model.load_model()
        after = [o.product_ids for o in recommend_outfits(prefs).outfits]
        assert before == after
