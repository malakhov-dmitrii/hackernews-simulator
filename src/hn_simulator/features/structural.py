"""Structural feature extraction from preprocessed HN stories."""
from __future__ import annotations

import re

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


def extract_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all structural feature extractors into a single numeric DataFrame.

    Returns only the computed feature columns (13 total), preserving index.
    """
    parts = [
        extract_title_features(df),
        extract_temporal_features(df),
        extract_url_features(df),
        extract_text_presence_features(df),
    ]
    return pd.concat(parts, axis=1)
