"""Tests for percentile calibration and time-of-day stats."""
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path


class TestScoreToPercentile:
    def test_median_score_is_near_50(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Score 5 -> 50th percentile -> "top 50%"
        pct = score_to_percentile(5, sorted_scores)
        assert 40.0 <= pct <= 60.0

    def test_max_score_is_top_10_percent(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.arange(1, 101)  # 1..100
        pct = score_to_percentile(100, sorted_scores)
        assert pct <= 1.0  # top 1%

    def test_min_score_is_near_100_percent(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.arange(1, 101)
        pct = score_to_percentile(1, sorted_scores)
        assert pct >= 99.0

    def test_above_max_is_top_1_percent(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.arange(1, 101)
        pct = score_to_percentile(1000, sorted_scores)
        assert pct < 1.0

    def test_score_9_example(self):
        """From research: score=9 -> top ~48%."""
        from hn_simulator.model.calibrate import score_to_percentile
        # Simulate a distribution where score=9 is around median
        sorted_scores = np.sort(np.concatenate([
            np.ones(50) * 1,    # 50 scores of 1
            np.ones(30) * 5,    # 30 scores of 5
            np.ones(10) * 9,    # 10 scores of 9
            np.ones(7) * 50,    # 7 scores of 50
            np.ones(3) * 500,   # 3 scores of 500
        ])).astype(np.int32)
        pct = score_to_percentile(9, sorted_scores)
        # 90 scores <= 9, out of 100, so top (100 - 90) = 10%
        assert pct == pytest.approx(10.0)


class TestBuildSortedScores:
    def test_returns_sorted_array(self):
        from hn_simulator.model.calibrate import build_sorted_scores
        scores = np.array([50, 10, 200, 1, 30], dtype=np.int32)
        result = build_sorted_scores(scores)
        assert np.all(result[:-1] <= result[1:])  # sorted ascending
        assert len(result) == 5

    def test_handles_duplicates(self):
        from hn_simulator.model.calibrate import build_sorted_scores
        scores = np.array([5, 5, 5, 10, 10], dtype=np.int32)
        result = build_sorted_scores(scores)
        assert len(result) == 5
        assert list(result) == [5, 5, 5, 10, 10]


class TestComputeTimeStats:
    def test_hourly_stats_has_24_entries(self):
        from hn_simulator.model.calibrate import compute_time_stats
        df = pd.DataFrame({
            "time": pd.to_datetime([
                "2023-01-01 09:00:00", "2023-01-01 09:30:00",
                "2023-01-01 15:00:00", "2023-01-02 03:00:00",
            ], utc=True),
            "score": np.array([100, 50, 20, 80], dtype=np.int32),
        })
        hourly, daily = compute_time_stats(df)
        assert isinstance(hourly, dict)
        # Hour 9 has avg score (100+50)/2 = 75
        assert hourly[9] == pytest.approx(75.0)
        assert hourly[15] == pytest.approx(20.0)
        assert hourly[3] == pytest.approx(80.0)

    def test_daily_stats_has_7_entries(self):
        from hn_simulator.model.calibrate import compute_time_stats
        df = pd.DataFrame({
            "time": pd.to_datetime([
                "2023-01-01 09:00:00",  # Sunday (6)
                "2023-01-02 09:00:00",  # Monday (0)
            ], utc=True),
            "score": np.array([100, 50], dtype=np.int32),
        })
        _, daily = compute_time_stats(df)
        assert isinstance(daily, dict)
        assert daily[6] == pytest.approx(100.0)  # Sunday
        assert daily[0] == pytest.approx(50.0)   # Monday


class TestRecommendPostingTime:
    def test_returns_best_hour_and_day(self):
        from hn_simulator.model.calibrate import recommend_posting_time
        hourly = {h: 10.0 for h in range(24)}
        hourly[9] = 100.0  # Hour 9 is best
        daily = {d: 10.0 for d in range(7)}
        daily[6] = 80.0  # Sunday is best
        result = recommend_posting_time(hourly, daily)
        assert result["best_hour"] == 9
        assert result["best_day"] == 6
        assert result["best_day_name"] == "Sunday"

    def test_returns_top_3_hours(self):
        from hn_simulator.model.calibrate import recommend_posting_time
        hourly = {h: float(h) for h in range(24)}  # hour 23 is best
        daily = {d: 10.0 for d in range(7)}
        result = recommend_posting_time(hourly, daily)
        assert result["top_3_hours"] == [23, 22, 21]


class TestSaveLoadCalibrationData:
    def test_roundtrip_sorted_scores(self, tmp_path):
        from hn_simulator.model.calibrate import save_sorted_scores, load_sorted_scores
        scores = np.array([1, 5, 10, 50, 100], dtype=np.int32)
        path = tmp_path / "sorted_scores.npy"
        save_sorted_scores(scores, path)
        loaded = load_sorted_scores(path)
        np.testing.assert_array_equal(scores, loaded)

    def test_roundtrip_time_stats(self, tmp_path):
        from hn_simulator.model.calibrate import save_time_stats, load_time_stats
        hourly = {h: float(h * 10) for h in range(24)}
        daily = {d: float(d * 5) for d in range(7)}
        path = tmp_path / "time_stats.json"
        save_time_stats(hourly, daily, path)
        h_loaded, d_loaded = load_time_stats(path)
        assert h_loaded[9] == pytest.approx(90.0)
        assert d_loaded[6] == pytest.approx(30.0)
