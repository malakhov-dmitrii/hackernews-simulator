"""Train score and comment-count models from processed features."""
import json

import click
import numpy as np
import pandas as pd

from hn_simulator.config import MODELS_DIR, PROCESSED_DIR, TRAIN_CUTOFF_DATE, ensure_dirs
from hn_simulator.model.train import save_model, train_comment_count_model, train_score_model


@click.command()
@click.option("--features-dir", default=None, help="Directory with processed features (default: PROCESSED_DIR)")
@click.option("--models-dir", default=None, help="Directory to save models (default: MODELS_DIR)")
@click.option("--cutoff", default=TRAIN_CUTOFF_DATE, help="Temporal split cutoff date (YYYY-MM-DD)")
def main(features_dir, models_dir, cutoff):
    ensure_dirs()

    src = features_dir or str(PROCESSED_DIR)
    dst = models_dir or str(MODELS_DIR)

    click.echo(f"Loading features from {src}...")
    X = np.load(f"{src}/features.npy")
    y_score = np.load(f"{src}/labels_score.npy")
    y_comments = np.load(f"{src}/labels_comments.npy")
    with open(f"{src}/feature_names.json") as f:
        feature_names = json.load(f)

    click.echo(f"Feature matrix: {X.shape}, cutoff: {cutoff}")

    # Temporal split — load raw stories to get time column for splitting indices
    # We use a simple index-based split as a fallback when no time metadata exists
    # If stories.parquet exists alongside features, use temporal split properly
    from pathlib import Path
    from hn_simulator.config import RAW_DIR
    from hn_simulator.data.preprocess import preprocess_stories
    from hn_simulator.model.train import temporal_split

    raw_path = RAW_DIR / "stories.parquet"
    if raw_path.exists():
        click.echo("Loading stories for temporal split...")
        df = pd.read_parquet(raw_path)
        df = preprocess_stories(df)
        train_df, val_df = temporal_split(df, cutoff)
        train_idx = train_df.index
        val_idx = val_df.index

        # Re-index to positional after preprocess (df index may not be 0-based)
        df_reset = df.reset_index(drop=True)
        train_df_reset, val_df_reset = temporal_split(df_reset, cutoff)
        train_pos = train_df_reset.index.tolist()
        val_pos = val_df_reset.index.tolist()

        X_train, X_val = X[train_pos], X[val_pos]
        y_score_train, y_score_val = y_score[train_pos], y_score[val_pos]
        y_comments_train, y_comments_val = y_comments[train_pos], y_comments[val_pos]
        click.echo(f"Temporal split: {len(train_pos)} train, {len(val_pos)} val")
    else:
        # Fallback: 80/20 split by position
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_score_train, y_score_val = y_score[:split], y_score[split:]
        y_comments_train, y_comments_val = y_comments[:split], y_comments[split:]
        click.echo(f"Positional split (no raw parquet): {split} train, {n - split} val")

    click.echo("Training score model...")
    score_model, score_metrics = train_score_model(
        X_train, y_score_train, X_val, y_score_val, feature_names
    )
    click.echo(f"Score model — val_rmse={score_metrics['val_rmse']:.4f}, val_mae={score_metrics['val_mae']:.4f}")

    click.echo("Training comment-count model...")
    comment_model, comment_metrics = train_comment_count_model(
        X_train, y_comments_train, X_val, y_comments_val, feature_names
    )
    click.echo(f"Comment model — val_rmse={comment_metrics['val_rmse']:.4f}, val_mae={comment_metrics['val_mae']:.4f}")

    from pathlib import Path as _Path
    dst_path = _Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    score_path = dst_path / "score_model.txt"
    comment_path = dst_path / "comment_model.txt"
    save_model(score_model, score_path)
    save_model(comment_model, comment_path)
    click.echo(f"Saved score model -> {score_path}")
    click.echo(f"Saved comment model -> {comment_path}")


if __name__ == "__main__":
    main()
