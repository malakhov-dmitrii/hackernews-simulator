"""Tests for prediction module.
All synthetic data uses fixed seed via np.random.default_rng(42).
"""
import pytest
import numpy as np
import lightgbm as lgb


@pytest.fixture
def synthetic_training_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 10 + X[:, 1] * 5 + rng.standard_normal(n) * 2).clip(0, None)
    feature_names = [f"feat_{i}" for i in range(20)]
    return X, y, feature_names


@pytest.fixture
def trained_model(synthetic_training_data):
    """Train a quick model for prediction tests."""
    from hackernews_simulator.model.train import train_score_model
    X, y, names = synthetic_training_data
    model, _ = train_score_model(X[:400], y[:400], X[400:], y[400:], names)
    return model


class TestPredict:
    def test_predict_score_returns_float(self, trained_model):
        from hackernews_simulator.model.predict import predict_score
        rng = np.random.default_rng(99)
        X = rng.standard_normal((1, 20)).astype(np.float32)
        score = predict_score(trained_model, X)
        assert isinstance(score, float)
        assert score >= 0  # scores can't be negative

    def test_predict_score_batch(self, trained_model):
        from hackernews_simulator.model.predict import predict_score
        rng = np.random.default_rng(99)
        X = rng.standard_normal((5, 20)).astype(np.float32)
        scores = predict_score(trained_model, X, batch=True)
        assert len(scores) == 5
        assert all(s >= 0 for s in scores)

    def test_predict_clamps_negative(self, trained_model):
        from hackernews_simulator.model.predict import predict_score
        # Extreme negative features might produce negative raw prediction
        X = np.full((1, 20), -100.0, dtype=np.float32)
        score = predict_score(trained_model, X)
        assert score >= 0


class TestPredictionResult:
    def test_prediction_result_dataclass(self):
        from hackernews_simulator.model.predict import PredictionResult
        result = PredictionResult(
            predicted_score=42.5,
            predicted_comments=15.3,
            reception_label="hot",
            confidence=0.75,
            label_distribution={"flop": 0.05, "moderate": 0.15, "hot": 0.55, "viral": 0.25},
        )
        assert result.predicted_score == 42.5
        assert result.reception_label == "hot"
