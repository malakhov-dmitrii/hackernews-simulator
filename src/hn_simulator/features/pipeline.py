"""Feature pipeline combining structural features and title embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hn_simulator.features.structural import extract_structural_features
from hn_simulator.features.text import extract_title_embeddings


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from a preprocessed stories DataFrame.

    Concatenates structural features (13 cols) and title embeddings (384 dims).
    Returns (X, feature_names) where X is float32 ndarray of shape (n, 397).
    Any NaN values are filled with 0.
    """
    structural_df = extract_structural_features(df)
    structural_names = list(structural_df.columns)

    embeddings = extract_title_embeddings(df)  # (n, 384)
    emb_names = [f"emb_{i}" for i in range(embeddings.shape[1])]

    structural_arr = structural_df.to_numpy(dtype=np.float32)
    X = np.concatenate([structural_arr, embeddings], axis=1)

    # Fill NaN with 0
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = 0.0

    feature_names = structural_names + emb_names
    return X, feature_names


def build_feature_matrix_for_input(
    title: str, description: str = "", url: str = ""
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix for a single prediction input.

    Constructs a single-row DataFrame with open-index schema columns,
    runs it through preprocess_stories and build_feature_matrix.
    """
    from hn_simulator.data.preprocess import preprocess_stories

    df = pd.DataFrame(
        {
            "id": np.array([0], dtype=np.uint32),
            "type": np.array([1], dtype=np.int8),
            "title": [title],
            "url": [url],
            "text": [description],
            "score": np.array([0], dtype=np.int32),
            "descendants": np.array([0], dtype=np.int32),
            "by": ["unknown"],
            "time": [pd.Timestamp.now(tz="UTC")],
            "dead": np.array([0], dtype=np.uint8),
            "deleted": np.array([0], dtype=np.uint8),
        }
    )
    df = preprocess_stories(df)
    return build_feature_matrix(df)
