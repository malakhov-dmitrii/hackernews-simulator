"""Tests for 5-class multiclass classification system."""
from __future__ import annotations

import numpy as np
import pytest


class TestScoreToClassLabel:
    """Boundary value tests for score_to_class_label."""

    def test_score_0_is_flop(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(0) == "flop"

    def test_negative_score_is_flop(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(-5) == "flop"

    def test_score_3_is_flop(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(3) == "flop"

    def test_score_4_is_low(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(4) == "low"

    def test_score_15_is_low(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(15) == "low"

    def test_score_16_is_moderate(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(16) == "moderate"

    def test_score_100_is_moderate(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(100) == "moderate"

    def test_score_101_is_hot(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(101) == "hot"

    def test_score_300_is_hot(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(300) == "hot"

    def test_score_301_is_viral(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(301) == "viral"

    def test_score_1000_is_viral(self):
        from hackernews_simulator.model.labels import score_to_class_label
        assert score_to_class_label(1000) == "viral"


class TestScoreToClassIndex:
    """Tests for score_to_class_index returning 0-4."""

    def test_flop_index_0(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(1) == 0

    def test_low_index_1(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(10) == 1

    def test_moderate_index_2(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(50) == 2

    def test_hot_index_3(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(200) == 3

    def test_viral_index_4(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(500) == 4

    def test_boundary_score_3_is_0(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(3) == 0

    def test_boundary_score_4_is_1(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(4) == 1

    def test_boundary_score_301_is_4(self):
        from hackernews_simulator.model.labels import score_to_class_index
        assert score_to_class_index(301) == 4


class TestExpectedScoreFromProbs:
    """Tests for expected_score_from_probs dot product."""

    def test_certain_flop(self):
        from hackernews_simulator.model.labels import expected_score_from_probs, BUCKET_MEDIANS
        probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = expected_score_from_probs(probs)
        assert abs(result - BUCKET_MEDIANS[0]) < 1e-6

    def test_certain_viral(self):
        from hackernews_simulator.model.labels import expected_score_from_probs, BUCKET_MEDIANS
        probs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        result = expected_score_from_probs(probs)
        assert abs(result - BUCKET_MEDIANS[4]) < 1e-6

    def test_uniform_probs(self):
        from hackernews_simulator.model.labels import expected_score_from_probs, BUCKET_MEDIANS
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        expected = float(np.dot(probs, BUCKET_MEDIANS))
        result = expected_score_from_probs(probs)
        assert abs(result - expected) < 1e-6

    def test_custom_bucket_medians(self):
        from hackernews_simulator.model.labels import expected_score_from_probs
        probs = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        medians = np.array([2.0, 8.0, 50.0, 200.0, 500.0])
        result = expected_score_from_probs(probs, medians)
        assert abs(result - 5.0) < 1e-6

    def test_bucket_medians_values(self):
        from hackernews_simulator.model.labels import BUCKET_MEDIANS
        assert len(BUCKET_MEDIANS) == 5
        # Medians must be strictly increasing
        assert all(BUCKET_MEDIANS[i] < BUCKET_MEDIANS[i + 1] for i in range(4))


class TestTrainMulticlassModel:
    """Tests for train_multiclass_model function."""

    @pytest.fixture
    def small_dataset(self):
        """Tiny synthetic dataset with all 5 classes present."""
        rng = np.random.default_rng(42)
        n_train, n_val, n_features = 200, 50, 10
        X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
        X_val = rng.standard_normal((n_val, n_features)).astype(np.float32)
        # Ensure all 5 classes appear in both splits
        y_train = np.tile(np.arange(5), n_train // 5).astype(np.int32)
        y_val = np.tile(np.arange(5), n_val // 5).astype(np.int32)
        feature_names = [f"f{i}" for i in range(n_features)]
        return X_train, y_train, X_val, y_val, feature_names

    def test_returns_booster_and_metrics(self, small_dataset):
        from hackernews_simulator.model.train import train_multiclass_model
        X_train, y_train, X_val, y_val, feature_names = small_dataset
        model, metrics = train_multiclass_model(X_train, y_train, X_val, y_val, feature_names)
        import lightgbm as lgb
        assert isinstance(model, lgb.Booster)
        assert "val_accuracy" in metrics
        assert "val_logloss" in metrics

    def test_predict_shape_n_by_5(self, small_dataset):
        from hackernews_simulator.model.train import train_multiclass_model
        X_train, y_train, X_val, y_val, feature_names = small_dataset
        model, _ = train_multiclass_model(X_train, y_train, X_val, y_val, feature_names)
        probs = model.predict(X_val)
        assert probs.shape == (len(X_val), 5), f"Expected ({len(X_val)}, 5), got {probs.shape}"

    def test_predict_probs_sum_to_one(self, small_dataset):
        from hackernews_simulator.model.train import train_multiclass_model
        X_train, y_train, X_val, y_val, feature_names = small_dataset
        model, _ = train_multiclass_model(X_train, y_train, X_val, y_val, feature_names)
        probs = model.predict(X_val)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_val_accuracy_in_range(self, small_dataset):
        from hackernews_simulator.model.train import train_multiclass_model
        X_train, y_train, X_val, y_val, feature_names = small_dataset
        _, metrics = train_multiclass_model(X_train, y_train, X_val, y_val, feature_names)
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_val_logloss_positive(self, small_dataset):
        from hackernews_simulator.model.train import train_multiclass_model
        X_train, y_train, X_val, y_val, feature_names = small_dataset
        _, metrics = train_multiclass_model(X_train, y_train, X_val, y_val, feature_names)
        assert metrics["val_logloss"] > 0.0

    def test_class_weights_computed(self, small_dataset):
        """Class weights via compute_class_weight('balanced') are applied without error."""
        from hackernews_simulator.model.train import train_multiclass_model
        from sklearn.utils.class_weight import compute_class_weight
        X_train, y_train, X_val, y_val, feature_names = small_dataset
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        assert len(weights) == 5
        # balanced weights for equal class sizes should all be equal
        assert np.allclose(weights, weights[0], atol=1e-6)
        # Model still trains without error
        model, metrics = train_multiclass_model(X_train, y_train, X_val, y_val, feature_names)
        assert model is not None
