"""LightGBM model training with temporal split."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from hackernews_simulator.config import LIGHTGBM_PARAMS, LIGHTGBM_MULTICLASS_PARAMS


def temporal_split(
    df: pd.DataFrame, cutoff: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by time column at cutoff date.

    Args:
        df: DataFrame with 'time' column (timestamp, UTC).
        cutoff: Date string like "2023-01-01". Train < cutoff, val >= cutoff.

    Returns:
        (train_df, val_df) — disjoint sets with no data leakage.
    """
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train_mask = df["time"] < cutoff_ts
    train_df = df[train_mask].copy()
    val_df = df[~train_mask].copy()
    return train_df, val_df


def _train_regression_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    params: dict,
) -> tuple[lgb.Booster, dict]:
    """Shared training logic for regression models.

    Applies log1p to targets before training, computes metrics in original scale.

    Returns:
        (booster, {"val_rmse": float, "val_mae": float})
    """
    # Apply log1p transform to targets
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    train_dataset = lgb.Dataset(
        X_train, label=y_train_log, feature_name=feature_names
    )
    val_dataset = lgb.Dataset(
        X_val, label=y_val_log, feature_name=feature_names, reference=train_dataset
    )

    # Separate n_estimators and early_stopping_rounds from params
    train_params = {k: v for k, v in params.items()
                    if k not in ("n_estimators", "early_stopping_rounds")}
    n_estimators = params.get("n_estimators", 500)
    early_stopping_rounds = params.get("early_stopping_rounds", 50)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        train_params,
        train_dataset,
        num_boost_round=n_estimators,
        valid_sets=[val_dataset],
        callbacks=callbacks,
    )

    # Predict in log space, inverse transform to original scale
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log).clip(0, None)

    val_rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
    val_mae = float(np.mean(np.abs(preds - y_val)))

    return model, {"val_rmse": val_rmse, "val_mae": val_mae}


def train_score_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM model to predict story score.

    Returns:
        (booster, {"val_rmse": float, "val_mae": float}) — metrics in original scale.
    """
    return _train_regression_model(
        X_train, y_train, X_val, y_val, feature_names, LIGHTGBM_PARAMS
    )


def train_comment_count_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM model to predict comment count (descendants).

    Returns:
        (booster, {"val_rmse": float, "val_mae": float}) — metrics in original scale.
    """
    return _train_regression_model(
        X_train, y_train, X_val, y_val, feature_names, LIGHTGBM_PARAMS
    )


def train_multiclass_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM multiclass model to predict 5-class reception label.

    Classes: 0=flop, 1=low, 2=moderate, 3=hot, 4=viral.
    Uses balanced class weights to handle class imbalance.

    Args:
        X_train: Training feature matrix of shape (n_train, n_features).
        y_train: Integer class labels 0-4 for training set.
        X_val: Validation feature matrix of shape (n_val, n_features).
        y_val: Integer class labels 0-4 for validation set.
        feature_names: List of feature name strings.

    Returns:
        (booster, {"val_accuracy": float, "val_logloss": float})
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weights = np.array([weights[int(c)] for c in y_train])

    train_dataset = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        weight=sample_weights,
    )
    val_dataset = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_names, reference=train_dataset
    )

    train_params = {k: v for k, v in LIGHTGBM_MULTICLASS_PARAMS.items()
                    if k not in ("n_estimators", "early_stopping_rounds")}
    n_estimators = LIGHTGBM_MULTICLASS_PARAMS.get("n_estimators", 500)
    early_stopping_rounds = LIGHTGBM_MULTICLASS_PARAMS.get("early_stopping_rounds", 50)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        train_params,
        train_dataset,
        num_boost_round=n_estimators,
        valid_sets=[val_dataset],
        callbacks=callbacks,
    )

    # Predict probabilities: shape (n_val, 5)
    probs = model.predict(X_val)
    preds = np.argmax(probs, axis=1)

    val_accuracy = float(np.mean(preds == y_val))

    # Compute log loss manually for the true classes
    eps = 1e-15
    true_probs = probs[np.arange(len(y_val)), y_val.astype(int)].clip(eps, 1 - eps)
    val_logloss = float(-np.mean(np.log(true_probs)))

    return model, {"val_accuracy": val_accuracy, "val_logloss": val_logloss}


def save_model(model: lgb.Booster, path: Path) -> None:
    """Save a LightGBM Booster to disk."""
    model.save_model(str(path))


def load_model(path: Path) -> lgb.Booster:
    """Load a LightGBM Booster from disk.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return lgb.Booster(model_file=str(path))
