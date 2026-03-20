"""SHAP-based feature importance explanations."""
from __future__ import annotations
import numpy as np


def explain_prediction(
    model,
    X_single: np.ndarray,
    feature_names: list[str],
    structural_names: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Explain a single prediction using SHAP TreeExplainer.

    Args:
        model: Trained LightGBM Booster.
        X_single: Feature matrix of shape (1, n_features).
        feature_names: List of feature name strings.
        structural_names: If provided, restrict output to these features only.
        top_k: Maximum number of features to return.

    Returns:
        List of dicts with keys "feature", "importance" (signed SHAP value),
        "direction" ("up" or "down"), sorted by absolute importance descending.
    """
    import shap  # lazy import

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)

    # Handle multiclass: shape (1, n_features, n_classes) or list of arrays
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (1, n_features, n_classes) — average across classes
        avg_shap = np.mean(np.abs(shap_values[0]), axis=-1)
        signed_shap = np.mean(shap_values[0], axis=-1)
    elif isinstance(shap_values, list):
        # list of (1, n_features) arrays, one per class
        stacked = np.stack(shap_values)  # (n_classes, 1, n_features)
        avg_shap = np.mean(np.abs(stacked[:, 0, :]), axis=0)
        signed_shap = np.mean(stacked[:, 0, :], axis=0)
    else:
        # Regression: shape (1, n_features)
        avg_shap = np.abs(shap_values[0])
        signed_shap = shap_values[0]

    # Filter to structural features if requested
    if structural_names is not None:
        mask = [i for i, name in enumerate(feature_names) if name in structural_names]
    else:
        mask = list(range(len(feature_names)))

    # Sort by absolute importance
    masked_importance = [
        (mask[j], avg_shap[mask[j]], signed_shap[mask[j]]) for j in range(len(mask))
    ]
    masked_importance.sort(key=lambda x: x[1], reverse=True)

    result = []
    for idx, abs_imp, signed_imp in masked_importance[:top_k]:
        result.append({
            "feature": feature_names[idx],
            "importance": float(signed_imp),
            "direction": "up" if signed_imp > 0 else "down",
        })
    return result


def format_explanation(features: list[dict]) -> str:
    """Format a list of feature explanation dicts into a human-readable string.

    Args:
        features: List of dicts from explain_prediction.

    Returns:
        Multi-line string with arrows and feature names, or "" if empty.
    """
    if not features:
        return ""
    lines = []
    for f in features:
        arrow = "↑" if f["direction"] == "up" else "↓"
        lines.append(f"  {arrow} {f['feature']} ({f['importance']:+.2f})")
    return "\n".join(lines)
