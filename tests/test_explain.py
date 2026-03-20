"""Tests for SHAP explanations."""
import pytest
import numpy as np


@pytest.fixture
def trained_model_and_data():
    import lightgbm as lgb
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = (X[:, 0] * 5 + X[:, 1] * 3 + rng.standard_normal(300)).clip(0, None)
    names = [f"struct_{i}" for i in range(5)] + [f"emb_{i}" for i in range(15)]

    ds = lgb.Dataset(X[:240], np.log1p(y[:240]), feature_name=names)
    val_ds = lgb.Dataset(X[240:], np.log1p(y[240:]), reference=ds)
    params = {"objective": "regression", "metric": "rmse", "num_leaves": 15, "verbose": -1}
    model = lgb.train(params, ds, num_boost_round=50, valid_sets=[val_ds],
                      callbacks=[lgb.log_evaluation(0)])
    return model, X[:1], names


class TestExplainPrediction:
    def test_returns_list(self, trained_model_and_data):
        from hackernews_simulator.model.explain import explain_prediction
        model, X, names = trained_model_and_data
        result = explain_prediction(model, X, names, top_k=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_sorted_by_importance(self, trained_model_and_data):
        from hackernews_simulator.model.explain import explain_prediction
        model, X, names = trained_model_and_data
        result = explain_prediction(model, X, names, top_k=5)
        importances = [abs(r["importance"]) for r in result]
        assert importances == sorted(importances, reverse=True)

    def test_has_direction(self, trained_model_and_data):
        from hackernews_simulator.model.explain import explain_prediction
        model, X, names = trained_model_and_data
        result = explain_prediction(model, X, names, top_k=3)
        for r in result:
            assert r["direction"] in ("up", "down")

    def test_structural_only_filter(self, trained_model_and_data):
        from hackernews_simulator.model.explain import explain_prediction
        model, X, names = trained_model_and_data
        structural = [f"struct_{i}" for i in range(5)]
        result = explain_prediction(model, X, names, structural_names=structural, top_k=5)
        for r in result:
            assert r["feature"] in structural

    def test_returns_fewer_if_not_enough(self, trained_model_and_data):
        from hackernews_simulator.model.explain import explain_prediction
        model, X, names = trained_model_and_data
        structural = [f"struct_{i}" for i in range(2)]
        result = explain_prediction(model, X, names, structural_names=structural, top_k=5)
        assert len(result) <= 2


class TestFormatExplanation:
    def test_format_string(self):
        from hackernews_simulator.model.explain import format_explanation
        features = [
            {"feature": "is_show_hn", "importance": 2.3, "direction": "up"},
            {"feature": "title_length", "importance": -0.9, "direction": "down"},
        ]
        text = format_explanation(features)
        assert "is_show_hn" in text
        assert "title_length" in text

    def test_empty_list(self):
        from hackernews_simulator.model.explain import format_explanation
        assert format_explanation([]) == ""
