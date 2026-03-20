"""Rule-based reception label classification."""
from __future__ import annotations

import numpy as np

from hackernews_simulator.config import SCORE_THRESHOLDS

# Ordered label sequence for distribution computation (5-class system)
_LABELS = ("flop", "low", "moderate", "hot", "viral")

# Threshold boundaries in ascending order
_BOUNDARIES = (
    SCORE_THRESHOLDS["flop"],      # 3
    SCORE_THRESHOLDS["low"],       # 15
    SCORE_THRESHOLDS["moderate"],  # 100
    SCORE_THRESHOLDS["hot"],       # 300
)

_DESCRIPTIONS = {
    "flop": (
        "Post received minimal engagement. Score stayed very low, suggesting it "
        "did not resonate with the HN community or was posted at a bad time."
    ),
    "low": (
        "Post received below-average engagement. Attracted only a handful of upvotes "
        "and limited discussion — typical for posts that did not break through."
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

# Bucket medians for expected score computation (one per class)
BUCKET_MEDIANS = np.array([1.0, 7.0, 37.0, 159.0, 441.0])


def score_to_class_label(score: float) -> str:
    """Map a score to a 5-class label.

    Boundaries:
        score <= 3   -> "flop"
        score <= 15  -> "low"
        score <= 100 -> "moderate"
        score <= 300 -> "hot"
        score > 300  -> "viral"

    Args:
        score: Story score (predicted or actual).

    Returns:
        One of: "flop", "low", "moderate", "hot", "viral".
    """
    if score <= _BOUNDARIES[0]:
        return "flop"
    if score <= _BOUNDARIES[1]:
        return "low"
    if score <= _BOUNDARIES[2]:
        return "moderate"
    if score <= _BOUNDARIES[3]:
        return "hot"
    return "viral"


def score_to_class_index(score: float) -> int:
    """Map a score to a class index 0-4.

    Returns:
        0 = flop, 1 = low, 2 = moderate, 3 = hot, 4 = viral.
    """
    label = score_to_class_label(score)
    return _LABELS.index(label)


def expected_score_from_probs(
    probs: np.ndarray, bucket_medians: np.ndarray | None = None
) -> float:
    """Compute expected score as dot product of class probabilities and bucket medians.

    Args:
        probs: Array of shape (5,) with class probabilities summing to 1.
        bucket_medians: Optional array of shape (5,). Defaults to BUCKET_MEDIANS.

    Returns:
        Expected score (float).
    """
    if bucket_medians is None:
        bucket_medians = BUCKET_MEDIANS
    return float(np.dot(probs, bucket_medians))


def classify_reception(score: float, comment_count: float) -> str:
    """Classify a story's reception based on score.

    Thresholds (from SCORE_THRESHOLDS):
        score <= 3   -> "flop"
        score <= 15  -> "low"
        score <= 100 -> "moderate"
        score <= 300 -> "hot"
        score > 300  -> "viral"

    Args:
        score: Predicted or actual story score.
        comment_count: Predicted or actual comment count (available for future use).

    Returns:
        One of: "flop", "low", "moderate", "hot", "viral".
    """
    return score_to_class_label(score)


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

    # Bucket centres in log-score space (5 classes)
    centres = [
        np.log1p(1.0),    # flop centre ~0-3
        np.log1p(7.0),    # low centre ~4-15
        np.log1p(37.0),   # moderate centre ~16-100
        np.log1p(159.0),  # hot centre ~101-300
        np.log1p(441.0),  # viral centre ~301+
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
        label: One of "flop", "low", "moderate", "hot", "viral".

    Returns:
        Multi-sentence description of what the label means in HN context.
    """
    return _DESCRIPTIONS[label]
