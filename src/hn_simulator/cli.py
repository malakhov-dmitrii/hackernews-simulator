"""Click CLI entrypoint for HN Reaction Simulator."""
from __future__ import annotations

import json
import sys

import click

from hn_simulator.config import LANCEDB_DIR, MODELS_DIR, RAW_DIR, ensure_dirs
from hn_simulator.simulator import HNSimulator


def _bar(fraction: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar."""
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _human_output(title: str, result) -> None:
    """Print human-readable prediction output."""
    sep = "═" * 43
    click.echo(sep)
    click.echo("  HN Reaction Simulator")
    click.echo(sep)
    click.echo()
    click.echo(f"  Title: {title}")
    click.echo()
    click.echo(f"  Predicted Score:    ~{int(result.predicted_score)} points")
    click.echo(f"  Predicted Comments: ~{int(result.predicted_comments)}")
    click.echo(
        f"  Reception: {result.reception_label.upper()} ({int(result.confidence * 100)}% confidence)"
    )
    click.echo()
    if result.label_distribution:
        click.echo("  Label Distribution:")
        for label, prob in result.label_distribution.items():
            bar = _bar(prob)
            click.echo(f"    {label:<10}{bar} {int(prob * 100)}%")
        click.echo()
    if result.simulated_comments:
        click.echo("  --- Simulated Comments ---")
        click.echo()
        for comment in result.simulated_comments:
            author = comment.get("username", comment.get("author", "hn_user"))
            text = comment.get("comment", comment.get("text", ""))
            tone = comment.get("tone", "")
            tone_str = f" [{tone}]" if tone else ""
            click.echo(f"  {author}{tone_str}: {text}")
            click.echo()
        click.echo()
    click.echo(sep)


@click.group()
def main() -> None:
    """HN Reaction Simulator — predict how Hacker News will react to your post."""


@main.command()
@click.option("--title", required=True, help="Story title (e.g. 'Show HN: My Project')")
@click.option("--description", default="", help="Optional description / body text")
@click.option("--no-comments", "skip_comments", is_flag=True, default=False, help="Skip comment generation")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output JSON instead of human-readable text")
def predict(title: str, description: str, skip_comments: bool, output_json: bool) -> None:
    """Predict HN community reaction for a story."""
    score_model_path = MODELS_DIR / "score_model.txt"
    comment_model_path = MODELS_DIR / "comment_model.txt"

    try:
        simulator = HNSimulator(
            score_model_path=score_model_path,
            comment_model_path=comment_model_path,
            lancedb_path=LANCEDB_DIR,
        )
    except Exception as exc:
        click.echo(f"Error loading simulator: {exc}", err=True)
        sys.exit(1)

    result = simulator.simulate(
        title=title,
        description=description,
        generate_comments=not skip_comments,
    )

    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _human_output(title, result)


@main.command()
@click.option("--sample-size", default=100_000, show_default=True, help="Number of stories to fetch and train on")
@click.option("--min-score", default=1, show_default=True, help="Minimum story score to include")
def train(sample_size: int, min_score: int) -> None:
    """Run full training pipeline: fetch, preprocess, build features, train, save."""
    ensure_dirs()
    click.echo(f"Starting training pipeline (sample_size={sample_size}, min_score={min_score})...")

    click.echo("Step 1/4: Fetching data...")
    from hn_simulator.data.fetch import fetch_stories_stratified
    df = fetch_stories_stratified(total_limit=sample_size)
    stories_path = RAW_DIR / "stories.parquet"
    df.to_parquet(stories_path)
    click.echo(f"  Fetched {len(df)} stories -> {stories_path}")

    click.echo("Step 2/4: Preprocessing...")
    from hn_simulator.data.preprocess import preprocess_stories
    df = preprocess_stories(df)
    click.echo(f"  Preprocessed {len(df)} stories")

    click.echo("Step 3/4: Building feature matrix...")
    from hn_simulator.features.pipeline import build_feature_matrix
    from hn_simulator.config import PROCESSED_DIR
    import numpy as np
    X, feature_names = build_feature_matrix(df)
    np.save(PROCESSED_DIR / "features.npy", X)
    click.echo(f"  Features shape: {X.shape}")

    click.echo("Step 4/4: Training models...")
    from hn_simulator.model.train import train_model, save_model
    score_labels = df["score"].values
    comment_labels = df["descendants"].fillna(0).values
    score_model = train_model(X, score_labels)
    comment_model = train_model(X, comment_labels)
    save_model(score_model, MODELS_DIR / "score_model.txt")
    save_model(comment_model, MODELS_DIR / "comment_model.txt")
    click.echo("  Models saved.")

    click.echo("Training complete.")


@main.command()
@click.option("--sample-size", default=100_000, show_default=True, help="Number of stories to fetch")
@click.option("--output-dir", default=None, help="Output directory (default: RAW_DIR from config)")
def fetch(sample_size: int, output_dir: str | None) -> None:
    """Fetch a stratified sample of HN stories from HuggingFace."""
    ensure_dirs()
    from pathlib import Path
    out = Path(output_dir) if output_dir else RAW_DIR

    click.echo(f"Fetching {sample_size} stories from open-index/hacker-news...")
    from hn_simulator.data.fetch import fetch_stories_stratified
    df = fetch_stories_stratified(total_limit=sample_size)
    path = out / "stories.parquet"
    df.to_parquet(path)
    click.echo(f"Saved {len(df)} stories to {path}")


@main.command("build-index")
@click.option("--sample-size", default=300_000, show_default=True, help="Number of stories to index")
def build_index(sample_size: int) -> None:
    """Build LanceDB vector index from fetched stories."""
    ensure_dirs()
    stories_path = RAW_DIR / "stories.parquet"
    if not stories_path.exists():
        click.echo(f"Stories file not found: {stories_path}. Run 'fetch' first.", err=True)
        sys.exit(1)

    click.echo(f"Building LanceDB index (sample_size={sample_size})...")
    import pandas as pd
    from hn_simulator.rag.index import build_index as _build_index
    df = pd.read_parquet(stories_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    _build_index(df, LANCEDB_DIR)
    click.echo(f"Index built at {LANCEDB_DIR}")
