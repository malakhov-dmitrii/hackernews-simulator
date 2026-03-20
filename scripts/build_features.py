"""Build feature matrix and embeddings from raw stories parquet."""
import json

import click
import numpy as np
import pandas as pd

from hackernews_simulator.config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from hackernews_simulator.data.preprocess import preprocess_stories
from hackernews_simulator.features.pipeline import build_feature_matrix
from hackernews_simulator.features.text import extract_title_embeddings


@click.command()
@click.option("--input", "input_path", default=None, help="Path to raw stories.parquet (default: RAW_DIR/stories.parquet)")
@click.option("--min-score", default=0, help="Minimum score filter for stories")
def main(input_path, min_score):
    ensure_dirs()

    src = input_path or str(RAW_DIR / "stories.parquet")
    click.echo(f"Loading stories from {src}...")
    df = pd.read_parquet(src)
    click.echo(f"Loaded {len(df)} stories")

    click.echo("Preprocessing stories...")
    df = preprocess_stories(df, min_score=min_score)
    click.echo(f"After preprocessing: {len(df)} stories")

    click.echo("Building feature matrix...")
    X, feature_names = build_feature_matrix(df)

    y_score = df["score"].values.astype(np.float32)
    y_comments = df["descendants"].values.astype(np.float32)

    click.echo("Extracting title embeddings...")
    embeddings = extract_title_embeddings(df)

    features_path = PROCESSED_DIR / "features.npy"
    labels_score_path = PROCESSED_DIR / "labels_score.npy"
    labels_comments_path = PROCESSED_DIR / "labels_comments.npy"
    feature_names_path = PROCESSED_DIR / "feature_names.json"
    embeddings_path = PROCESSED_DIR / "embeddings.npy"

    np.save(features_path, X)
    np.save(labels_score_path, y_score)
    np.save(labels_comments_path, y_comments)
    np.save(embeddings_path, embeddings)
    feature_names_path.write_text(json.dumps(feature_names))

    click.echo(f"Saved features.npy        ({X.shape}) -> {features_path}")
    click.echo(f"Saved labels_score.npy    ({y_score.shape}) -> {labels_score_path}")
    click.echo(f"Saved labels_comments.npy ({y_comments.shape}) -> {labels_comments_path}")
    click.echo(f"Saved embeddings.npy      ({embeddings.shape}) -> {embeddings_path}")
    click.echo(f"Saved feature_names.json  ({len(feature_names)} names) -> {feature_names_path}")


if __name__ == "__main__":
    main()
