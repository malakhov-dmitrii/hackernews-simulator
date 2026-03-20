"""Tests for backtesting module."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_dataset():
    """Synthetic dataset with 500 samples, 10 features, scores spanning all 5 classes."""
    rng = np.random.default_rng(42)
    n, n_features = 500, 10
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    # Scores spread across all classes: flop(<=3), low(<=15), moderate(<=100), hot(<=300), viral(>300)
    # Use categorical distribution to ensure all classes present
    class_scores = [1.0, 8.0, 50.0, 150.0, 400.0]
    class_indices = rng.integers(0, 5, size=n)
    y = np.array([class_scores[i] + rng.uniform(-0.5, 0.5) for i in class_indices])
    feature_names = [f"f{i}" for i in range(n_features)]
    return X, y, feature_names


class TestSplitTrainTest:
    """Tests for split_train_test function."""

    def test_default_test_fraction(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import split_train_test
        X, y, _ = synthetic_dataset
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        n = len(X)
        assert len(X_test) == int(n * 0.2)
        assert len(X_train) == n - len(X_test)

    def test_split_sizes_sum_to_total(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import split_train_test
        X, y, _ = synthetic_dataset
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_split_is_deterministic_with_seed(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import split_train_test
        X, y, _ = synthetic_dataset
        result1 = split_train_test(X, y, seed=42)
        result2 = split_train_test(X, y, seed=42)
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])

    def test_different_seeds_give_different_splits(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import split_train_test
        X, y, _ = synthetic_dataset
        _, X_test_42, _, _ = split_train_test(X, y, seed=42)
        _, X_test_99, _, _ = split_train_test(X, y, seed=99)
        # Different seeds should yield different test sets
        assert not np.array_equal(X_test_42, X_test_99)

    def test_custom_test_fraction(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import split_train_test
        X, y, _ = synthetic_dataset
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_fraction=0.3)
        n = len(X)
        assert len(X_test) == int(n * 0.3)

    def test_y_shapes_match_X(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import split_train_test
        X, y, _ = synthetic_dataset
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_returns_dict(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        assert isinstance(results, dict)

    def test_accuracy_in_range(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        cm = results["confusion_matrix"]
        assert cm.shape == (5, 5)

    def test_confusion_matrix_nonnegative(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        assert (results["confusion_matrix"] >= 0).all()

    def test_spearman_rho_present(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        assert "spearman_rho" in results
        assert -1.0 <= results["spearman_rho"] <= 1.0

    def test_per_class_stats_present(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        assert "per_class" in results
        expected_classes = ["flop", "low", "moderate", "hot", "viral"]
        for cls in expected_classes:
            assert cls in results["per_class"]
            assert "precision" in results["per_class"][cls]
            assert "recall" in results["per_class"][cls]

    def test_per_class_precision_recall_in_range(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names)
        for cls, stats in results["per_class"].items():
            assert 0.0 <= stats["precision"] <= 1.0, f"{cls} precision out of range"
            assert 0.0 <= stats["recall"] <= 1.0, f"{cls} recall out of range"

    def test_result_is_deterministic(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        r1 = run_backtest(X, y, feature_names, seed=42)
        r2 = run_backtest(X, y, feature_names, seed=42)
        assert r1["accuracy"] == r2["accuracy"]
        np.testing.assert_array_equal(r1["confusion_matrix"], r2["confusion_matrix"])

    def test_confusion_matrix_row_sums_match_test_size(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        results = run_backtest(X, y, feature_names, test_fraction=0.2)
        # Rows represent true classes; total across all rows = test set size
        n_test = int(len(X) * 0.2)
        assert results["confusion_matrix"].sum() == n_test


class TestFormatBacktestReport:
    """Tests for format_backtest_report function."""

    @pytest.fixture
    def sample_results(self, synthetic_dataset):
        from hackernews_simulator.model.backtest import run_backtest
        X, y, feature_names = synthetic_dataset
        return run_backtest(X, y, feature_names)

    def test_returns_string(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert isinstance(report, str)

    def test_report_contains_accuracy(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert "Accuracy" in report

    def test_report_contains_spearman(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert "Spearman" in report or "spearman" in report.lower()

    def test_report_contains_per_class(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert "flop" in report
        assert "viral" in report

    def test_report_contains_confusion_matrix(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert "Confusion" in report or "confusion" in report.lower()

    def test_report_contains_header(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert "Backtest" in report

    def test_report_nonempty(self, sample_results):
        from hackernews_simulator.model.backtest import format_backtest_report
        report = format_backtest_report(sample_results)
        assert len(report) > 100
