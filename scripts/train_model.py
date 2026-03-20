"""Train score and comment-count models from processed features."""
import json

import click
import numpy as np
import pandas as pd

from hackernews_simulator.config import MODELS_DIR, PROCESSED_DIR, TRAIN_CUTOFF_DATE, ensure_dirs
from hackernews_simulator.model.train import save_model, train_comment_count_model, train_score_model


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
    from hackernews_simulator.config import RAW_DIR
    from hackernews_simulator.data.preprocess import preprocess_stories
    from hackernews_simulator.model.train import temporal_split

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

        # Fallback to 80/20 if temporal split gives empty train set
        if len(train_pos) == 0:
            click.echo("Temporal split empty — all data after cutoff. Falling back to 80/20 split.")
            n = len(X)
            split = int(n * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_score_train, y_score_val = y_score[:split], y_score[split:]
            y_comments_train, y_comments_val = y_comments[:split], y_comments[split:]
            click.echo(f"Positional split: {split} train, {n - split} val")
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

    # Train multiclass model
    click.echo("Training multiclass model...")
    from hackernews_simulator.model.train import train_multiclass_model
    from hackernews_simulator.model.labels import score_to_class_index
    y_class_train = np.array([score_to_class_index(s) for s in y_score_train], dtype=np.int32)
    y_class_val = np.array([score_to_class_index(s) for s in y_score_val], dtype=np.int32)
    multiclass_model, mc_metrics = train_multiclass_model(
        X_train, y_class_train, X_val, y_class_val, feature_names
    )
    mc_path = dst_path / "multiclass_model.txt"
    save_model(multiclass_model, mc_path)
    click.echo(f"Multiclass model — val_accuracy={mc_metrics['val_accuracy']:.4f}, val_logloss={mc_metrics['val_logloss']:.4f}")
    click.echo(f"Saved multiclass model -> {mc_path}")

    # Compute and save sorted_scores.npy for percentile calibration
    click.echo("Computing sorted scores for percentile calibration...")
    from hackernews_simulator.model.calibrate import build_sorted_scores, save_sorted_scores
    sorted_scores = build_sorted_scores(y_score)
    sorted_scores_path = _Path(src) / "sorted_scores.npy"
    save_sorted_scores(sorted_scores, sorted_scores_path)
    click.echo(f"Saved sorted scores -> {sorted_scores_path}")

    # Compute and save time_stats.json if raw stories exist
    from hackernews_simulator.config import RAW_DIR as _RAW_DIR
    raw_path = _RAW_DIR / "stories.parquet"
    if raw_path.exists():
        click.echo("Computing time stats...")
        import pandas as _pd
        from hackernews_simulator.model.calibrate import compute_time_stats, save_time_stats
        df_raw = _pd.read_parquet(raw_path)
        from hackernews_simulator.data.preprocess import preprocess_stories as _preprocess
        df_raw = _preprocess(df_raw)
        hourly, daily = compute_time_stats(df_raw)
        time_stats_path = _Path(src) / "time_stats.json"
        save_time_stats(hourly, daily, time_stats_path)
        click.echo(f"Saved time stats -> {time_stats_path}")

        # Compute and save domain_stats.json
        click.echo("Computing domain stats...")
        global_mean = float(df_raw["score"].mean())
        k = 10
        domain_groups = df_raw.groupby("domain")["score"]
        domain_stats = {}
        for domain, group in domain_groups:
            n = len(group)
            domain_mean = float(group.mean())
            smoothed = (n * domain_mean + k * global_mean) / (n + k)
            domain_stats[domain] = {"avg_score": smoothed, "post_count": n}
        domain_stats_path = _Path(src) / "domain_stats.json"
        import json as _json
        domain_stats_path.write_text(_json.dumps(domain_stats, indent=2))
        click.echo(f"Saved domain stats -> {domain_stats_path}")
    else:
        click.echo("No raw stories.parquet found — skipping time_stats and domain_stats.")


if __name__ == "__main__":
    main()
