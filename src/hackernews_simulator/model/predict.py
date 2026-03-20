"""Score and comment count prediction with inverse log1p transform."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np


@dataclass
class PredictionResult:
    """Result of a full prediction run for a single story pitch."""

    predicted_score: float
    predicted_comments: float
    reception_label: str
    confidence: float
    label_distribution: dict[str, float]


def predict_score(
    model: lgb.Booster,
    X: np.ndarray,
    batch: bool = False,
) -> float | list[float]:
    """Run model inference and inverse-transform the output.

    The model was trained on log1p(y), so predictions are in log space.
    This function applies expm1 to recover the original scale and clamps
    negative values to 0.

    Args:
        model: Trained LightGBM Booster.
        X: Feature matrix, shape (n_samples, n_features).
        batch: If True, return list[float] for all rows. If False (default),
               return a single float (uses the first row of X).

    Returns:
        Single float if batch=False, list of floats if batch=True.
        All values are >= 0.
    """
    raw = model.predict(X)
    scores = np.expm1(raw).clip(0, None)

    if batch:
        return scores.tolist()
    return float(scores[0])
