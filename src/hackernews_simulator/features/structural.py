"""Structural feature extraction from preprocessed HN stories."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def extract_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title-based features.

    Returns DataFrame with columns: title_length, title_word_count,
    title_has_question, title_has_number, is_show_hn, is_ask_hn.
    """
    titles = df["title"].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    result["title_length"] = titles.str.len().astype(int)
    result["title_word_count"] = titles.apply(
        lambda t: len(t.split()) if t else 0
    ).astype(int)
    result["title_has_question"] = titles.str.contains(r"\?", regex=True).astype(int)
    result["title_has_number"] = titles.str.contains(r"\d", regex=True).astype(int)
    result["is_show_hn"] = titles.str.startswith("Show HN:").astype(int)
    result["is_ask_hn"] = titles.str.startswith("Ask HN:").astype(int)
    return result


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the 'time' column (UTC timestamps).

    Returns DataFrame with columns: hour, day_of_week, is_weekend.
    day_of_week: 0=Monday … 6=Sunday; is_weekend: 1 if day_of_week >= 5.
    """
    times = df["time"]
    result = pd.DataFrame(index=df.index)
    result["hour"] = times.dt.hour.astype(int)
    result["day_of_week"] = times.dt.dayofweek.astype(int)
    result["is_weekend"] = (times.dt.dayofweek >= 5).astype(int)
    return result


def extract_url_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract URL-based features using the 'domain' column.

    Returns DataFrame with columns: has_url, is_github.
    """
    domain = df["domain"].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    result["has_url"] = (domain != "").astype(int)
    result["is_github"] = domain.str.contains("github.com", regex=False).astype(int)
    return result


def extract_text_presence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract text presence features from the 'clean_text' column.

    Returns DataFrame with columns: has_text, text_length.
    """
    clean_text = df["clean_text"].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    result["has_text"] = (clean_text != "").astype(int)
    result["text_length"] = clean_text.str.len().astype(int)
    return result


def compute_domain_stats(
    df: pd.DataFrame, k: int = 10, global_mean: float = 47.57
) -> dict:
    """Compute Bayesian-smoothed average score per domain.

    Formula: smoothed = (n * domain_mean + k * global_mean) / (n + k)

    Args:
        df: DataFrame with 'domain' and 'score' columns.
        k: Smoothing factor (default 10).
        global_mean: Global mean score used as prior (default 47.57).

    Returns:
        dict mapping domain -> {"avg_score": float, "post_count": int}
    """
    if df.empty:
        return {}
    grouped = df.groupby("domain")["score"]
    counts = grouped.count()
    means = grouped.mean()
    smoothed = (counts * means + k * global_mean) / (counts + k)
    result = {}
    for domain in counts.index:
        result[domain] = {
            "avg_score": float(smoothed[domain]),
            "post_count": int(counts[domain]),
        }
    return result


def extract_domain_reputation_features(
    df: pd.DataFrame, domain_stats: dict, global_mean: float = 47.57
) -> pd.DataFrame:
    """Extract domain reputation features using precomputed domain stats.

    Returns DataFrame with columns: domain_avg_score, domain_post_count.
    Unknown or empty domains fall back to global_mean and 0.
    """
    result = pd.DataFrame(index=df.index)
    domain_col = df["domain"].fillna("").astype(str)
    avg_scores = domain_col.map(
        lambda d: domain_stats[d]["avg_score"] if d in domain_stats else global_mean
    ).astype(float)
    post_counts = domain_col.map(
        lambda d: domain_stats[d]["post_count"] if d in domain_stats else 0
    ).astype(int)
    result["domain_avg_score"] = avg_scores
    result["domain_post_count"] = post_counts
    return result


def _load_domain_stats_from_disk() -> dict:
    """Load domain stats from the default processed data location if it exists."""
    try:
        from hackernews_simulator.config import PROCESSED_DIR
        path = Path(PROCESSED_DIR) / "domain_stats.json"
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def extract_structural_features(
    df: pd.DataFrame, domain_stats: dict | None = None
) -> pd.DataFrame:
    """Combine all structural feature extractors into a single numeric DataFrame.

    Returns only the computed feature columns (15 total), preserving index.
    If domain_stats is None, attempts to load from disk; falls back to empty dict.
    """
    if domain_stats is None:
        domain_stats = _load_domain_stats_from_disk()
    parts = [
        extract_title_features(df),
        extract_temporal_features(df),
        extract_url_features(df),
        extract_text_presence_features(df),
        extract_domain_reputation_features(df, domain_stats),
    ]
    return pd.concat(parts, axis=1)
