"""Tests for LightGBM model training.
All synthetic data uses fixed seed via np.random.default_rng(42).
"""
import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic training data with known patterns. Fixed seed."""
    rng = np.random.default_rng(42)
    n = 500
    # Feature 0 strongly correlates with target
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 10 + X[:, 1] * 5 + rng.standard_normal(n) * 2).clip(0, None)
    feature_names = [f"feat_{i}" for i in range(20)]
    return X, y, feature_names


class TestTemporalSplit:
    def test_split_by_date(self, sample_stories_df):
        from hackernews_simulator.model.train import temporal_split
        train_df, val_df = temporal_split(sample_stories_df, cutoff="2023-01-01")
        assert len(train_df) > 0
        assert len(val_df) > 0
        # Uses 'time' column (open-index schema)
        assert all(train_df["time"] < pd.Timestamp("2023-01-01", tz="UTC"))
        assert all(val_df["time"] >= pd.Timestamp("2023-01-01", tz="UTC"))

    def test_no_data_leak(self, sample_stories_df):
        from hackernews_simulator.model.train import temporal_split
        train_df, val_df = temporal_split(sample_stories_df, cutoff="2023-01-01")
        train_ids = set(train_df["id"])
        val_ids = set(val_df["id"])
        assert train_ids.isdisjoint(val_ids)


class TestTrainScoreModel:
    def test_returns_booster(self, synthetic_training_data):
        from hackernews_simulator.model.train import train_score_model
        X, y, names = synthetic_training_data
        model, metrics = train_score_model(
            X_train=X[:400], y_train=y[:400],
            X_val=X[400:], y_val=y[400:],
            feature_names=names,
        )
        assert isinstance(model, lgb.Booster)

    def test_returns_metrics_with_expected_keys(self, synthetic_training_data):
        from hackernews_simulator.model.train import train_score_model
        X, y, names = synthetic_training_data
        model, metrics = train_score_model(
            X_train=X[:400], y_train=y[:400],
            X_val=X[400:], y_val=y[400:],
            feature_names=names,
        )
        assert "val_rmse" in metrics
        assert "val_mae" in metrics
        assert metrics["val_rmse"] > 0

    def test_model_predicts_with_positive_correlation(self, synthetic_training_data):
        from hackernews_simulator.model.train import train_score_model
        X, y, names = synthetic_training_data
        model, _ = train_score_model(
            X_train=X[:400], y_train=y[:400],
            X_val=X[400:], y_val=y[400:],
            feature_names=names,
        )
        preds = model.predict(X[400:])
        correlation = np.corrcoef(preds, y[400:])[0, 1]
        assert correlation > 0.5


class TestTrainCommentModel:
    def test_returns_booster(self, synthetic_training_data):
        from hackernews_simulator.model.train import train_comment_count_model
        X, y, names = synthetic_training_data
        model, metrics = train_comment_count_model(
            X_train=X[:400], y_train=y[:400],
            X_val=X[400:], y_val=y[400:],
            feature_names=names,
        )
        assert isinstance(model, lgb.Booster)


class TestSaveLoadModel:
    def test_roundtrip_predictions_identical(self, synthetic_training_data, tmp_path):
        from hackernews_simulator.model.train import train_score_model, save_model, load_model
        X, y, names = synthetic_training_data
        model, _ = train_score_model(
            X_train=X[:400], y_train=y[:400],
            X_val=X[400:], y_val=y[400:],
            feature_names=names,
        )
        path = tmp_path / "model.txt"
        save_model(model, path)
        loaded = load_model(path)
        preds_original = model.predict(X[400:])
        preds_loaded = loaded.predict(X[400:])
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)

    def test_load_nonexistent_raises(self, tmp_path):
        from hackernews_simulator.model.train import load_model
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent_model.txt")
