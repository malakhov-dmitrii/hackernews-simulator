"""Backtesting module for multiclass reception classifier."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr

from hackernews_simulator.model.labels import score_to_class_index, BUCKET_MEDIANS
from hackernews_simulator.model.train import train_multiclass_model

_LABELS = ("flop", "low", "moderate", "hot", "viral")


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split feature matrix and target array into train/test sets.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        test_fraction: Fraction of data to use for the test set (default 0.2).
        seed: Random seed for reproducibility (default 42).

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_fraction, random_state=seed)


def run_backtest(
    X: np.ndarray,
    y_true_scores: np.ndarray,
    feature_names: list[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Run full backtest pipeline on the provided dataset.

    Steps:
        1. Split data into train/test using split_train_test.
        2. Convert scores to class indices (0-4).
        3. Train multiclass model on train split.
        4. Predict class probabilities on test split.
        5. Compute accuracy, confusion matrix, Spearman rho, per-class stats.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y_true_scores: Raw score targets of shape (n_samples,).
        feature_names: List of feature name strings.
        test_fraction: Fraction reserved for testing (default 0.2).
        seed: Random seed (default 42).

    Returns:
        Dict with keys:
            - "accuracy": float
            - "confusion_matrix": np.ndarray of shape (5, 5)
            - "spearman_rho": float
            - "per_class": dict mapping label -> {"precision": float, "recall": float}
    """
    # 1. Split
    X_train, X_test, y_scores_train, y_scores_test = split_train_test(
        X, y_true_scores, test_fraction=test_fraction, seed=seed
    )

    # 2. Convert scores to class indices
    y_train = np.array([score_to_class_index(s) for s in y_scores_train], dtype=np.int32)
    y_test = np.array([score_to_class_index(s) for s in y_scores_test], dtype=np.int32)

    # 3. Train multiclass model (use a small validation split from train)
    val_size = max(int(len(X_train) * 0.1), 5)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    model, _ = train_multiclass_model(X_tr, y_tr, X_val, y_val, feature_names)

    # 4. Predict on test set — shape (n_test, 5)
    probs = model.predict(X_test)
    y_pred = np.argmax(probs, axis=1)

    # 5. Compute metrics
    accuracy = float(np.mean(y_pred == y_test))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(5)))

    # Spearman rho: correlate expected score (from probs) vs actual score
    expected_scores = probs @ BUCKET_MEDIANS
    rho, _ = spearmanr(expected_scores, y_scores_test)
    spearman_rho = float(rho)

    # Per-class precision and recall
    per_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(_LABELS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[label] = {"precision": float(precision), "recall": float(recall)}

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "spearman_rho": spearman_rho,
        "per_class": per_class,
    }


def format_backtest_report(results: dict) -> str:
    """Format backtest results as an ASCII report.

    Args:
        results: Dict returned by run_backtest.

    Returns:
        Multi-line string report.
    """
    lines: list[str] = []
    lines.append("=== Backtest Report ===")
    lines.append(f"Accuracy: {results['accuracy'] * 100:.1f}%")
    lines.append(f"Spearman \u03c1: {results['spearman_rho']:.2f}")
    lines.append("")
    lines.append("Per-class:")
    for label in _LABELS:
        stats = results["per_class"][label]
        p = stats["precision"]
        r = stats["recall"]
        lines.append(f"  {label:<9} precision={p:.2f}  recall={r:.2f}")
    lines.append("")
    lines.append("Confusion Matrix:")
    header = "      " + "  ".join(f"{lbl[:3]:<5}" for lbl in _LABELS)
    lines.append(header)
    cm = results["confusion_matrix"]
    row_labels = ["flop", "low ", "mod ", "hot ", "vir "]
    for i, row_label in enumerate(row_labels):
        row_vals = "  ".join(f"{int(cm[i, j]):<5}" for j in range(5))
        lines.append(f"{row_label}  [{row_vals}]")
    return "\n".join(lines)
