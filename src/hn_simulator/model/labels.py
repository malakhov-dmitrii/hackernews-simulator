"""Rule-based reception label classification."""
from __future__ import annotations

import numpy as np

from hn_simulator.config import SCORE_THRESHOLDS

# Ordered label sequence for distribution computation
_LABELS = ("flop", "moderate", "hot", "viral")

# Threshold boundaries in ascending order
_BOUNDARIES = (
    SCORE_THRESHOLDS["flop"],      # 3
    SCORE_THRESHOLDS["moderate"],  # 15
    SCORE_THRESHOLDS["hot"],       # 100
)

_DESCRIPTIONS = {
    "flop": (
        "Post received minimal engagement. Score stayed very low, suggesting it "
        "did not resonate with the HN community or was posted at a bad time."
    ),
    "moderate": (
        "Post received moderate engagement. Attracted a small but engaged audience "
        "and generated some discussion — a typical outcome for niche or technical content."
    ),
    "hot": (
        "Post did well on HN. Reached the front page or came close, generating "
        "significant discussion and broad community interest."
    ),
    "viral": (
        "Post went viral on HN. Exceptional score and widespread discussion — "
        "one of the top posts of the day or week."
    ),
}


def classify_reception(score: float, comment_count: float) -> str:
    """Classify a story's reception based on score.

    Thresholds (from SCORE_THRESHOLDS):
        score <= 3   -> "flop"
        score <= 15  -> "moderate"
        score <= 100 -> "hot"
        score > 100  -> "viral"

    Args:
        score: Predicted or actual story score.
        comment_count: Predicted or actual comment count (available for future use).

    Returns:
        One of: "flop", "moderate", "hot", "viral".
    """
    if score <= _BOUNDARIES[0]:
        return "flop"
    if score <= _BOUNDARIES[1]:
        return "moderate"
    if score <= _BOUNDARIES[2]:
        return "hot"
    return "viral"


def classify_reception_with_confidence(
    predicted_score: float, predicted_comments: float
) -> tuple[str, float, dict[str, float]]:
    """Classify reception with a soft probability distribution over labels.

    Computes a softmax-like distribution based on distance from each threshold
    boundary, then returns the argmax label, its probability, and the full
    distribution.

    Args:
        predicted_score: Predicted story score.
        predicted_comments: Predicted comment count.

    Returns:
        (label, confidence, distribution) where:
            - label: winning label string
            - confidence: probability of the winning label, in (0, 1]
            - distribution: dict mapping each label to its probability (sums to ~1.0)
    """
    label = classify_reception(predicted_score, predicted_comments)

    # Build soft scores: higher = more likely. Use negative distance to each
    # bucket centre (in log-score space for numerical stability).
    log_score = np.log1p(max(predicted_score, 0.0))

    # Bucket centres in log-score space
    centres = [
        np.log1p(1.5),   # flop centre ~1-3
        np.log1p(9.0),   # moderate centre ~4-15
        np.log1p(57.5),  # hot centre ~16-100
        np.log1p(300.0), # viral centre ~101+
    ]

    # Negative squared distance → softmax gives probability
    dists = np.array([-(log_score - c) ** 2 for c in centres])
    # Temperature scaling: lower temperature = sharper distribution
    temperature = 0.5
    logits = dists / temperature
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    distribution = {lbl: float(p) for lbl, p in zip(_LABELS, probs)}
    confidence = float(distribution[label])

    return label, confidence, distribution


def get_reception_description(label: str) -> str:
    """Return a human-readable description of a reception label.

    Args:
        label: One of "flop", "moderate", "hot", "viral".

    Returns:
        Multi-sentence description of what the label means in HN context.
    """
    return _DESCRIPTIONS[label]
