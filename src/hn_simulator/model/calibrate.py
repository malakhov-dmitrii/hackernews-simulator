"""Percentile calibration and time-of-day analysis."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def score_to_percentile(score: float, sorted_scores: np.ndarray) -> float:
    """Map a score to 'top X%' using sorted historical scores.

    Args:
        score: Predicted score to rank.
        sorted_scores: Sorted ascending array of historical scores.

    Returns:
        Percentile as float (e.g., 10.0 means 'top 10%').
    """
    idx = np.searchsorted(sorted_scores, score, side="right")
    return 100.0 * (1.0 - idx / len(sorted_scores))


def build_sorted_scores(scores: np.ndarray) -> np.ndarray:
    """Sort score array ascending for percentile lookup.

    Args:
        scores: Array of historical scores (any order).

    Returns:
        New array sorted ascending.
    """
    return np.sort(scores)


def compute_time_stats(
    df: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute average score per hour (0-23) and per day-of-week (0=Mon..6=Sun).

    Args:
        df: DataFrame with 'time' (UTC timestamps) and 'score' columns.

    Returns:
        (hourly_stats, daily_stats) dicts mapping int keys to avg float scores.
    """
    df = df.copy()
    df["_hour"] = df["time"].dt.hour
    df["_dow"] = df["time"].dt.dayofweek
    hourly = df.groupby("_hour")["score"].mean().to_dict()
    daily = df.groupby("_dow")["score"].mean().to_dict()
    # Convert keys to int (groupby may produce int64)
    hourly = {int(k): float(v) for k, v in hourly.items()}
    daily = {int(k): float(v) for k, v in daily.items()}
    return hourly, daily


def recommend_posting_time(
    hourly: dict[int, float], daily: dict[int, float]
) -> dict:
    """Recommend best posting time based on hourly/daily stats.

    Args:
        hourly: Dict mapping hour (0-23) to avg score.
        daily: Dict mapping day-of-week (0=Mon..6=Sun) to avg score.

    Returns:
        Dict with best_hour, best_day, best_day_name, top_3_hours,
        best_hour_avg_score, best_day_avg_score.
    """
    best_hour = max(hourly, key=hourly.get)
    best_day = max(daily, key=daily.get)
    top_3_hours = sorted(hourly, key=hourly.get, reverse=True)[:3]
    return {
        "best_hour": best_hour,
        "best_day": best_day,
        "best_day_name": _DAY_NAMES[best_day],
        "top_3_hours": top_3_hours,
        "best_hour_avg_score": hourly[best_hour],
        "best_day_avg_score": daily[best_day],
    }


def save_sorted_scores(scores: np.ndarray, path: Path | str) -> None:
    """Save sorted scores array to .npy file."""
    np.save(str(path), scores)


def load_sorted_scores(path: Path | str) -> np.ndarray:
    """Load sorted scores array from .npy file."""
    return np.load(str(path))


def save_time_stats(
    hourly: dict[int, float], daily: dict[int, float], path: Path | str
) -> None:
    """Save hourly and daily stats to JSON file."""
    data = {
        "hourly": {str(k): v for k, v in hourly.items()},
        "daily": {str(k): v for k, v in daily.items()},
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_time_stats(
    path: Path | str,
) -> tuple[dict[int, float], dict[int, float]]:
    """Load hourly and daily stats from JSON file.

    Returns:
        (hourly, daily) dicts with int keys and float values.
    """
    data = json.loads(Path(path).read_text())
    hourly = {int(k): v for k, v in data["hourly"].items()}
    daily = {int(k): v for k, v in data["daily"].items()}
    return hourly, daily
