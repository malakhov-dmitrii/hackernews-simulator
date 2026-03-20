"""Click CLI entrypoint for HN Reaction Simulator."""
from __future__ import annotations

import json
import os
import sys
import warnings

# Suppress noisy warnings from dependencies before any imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MarkupResemblesLocator.*")
warnings.filterwarnings("ignore", message=".*table_names.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import click
import numpy as np

from hackernews_simulator.config import LANCEDB_DIR, MODELS_DIR, RAW_DIR, ensure_dirs
from hackernews_simulator.simulator import HNSimulator


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
    # Percentile and expected score (v2)
    if result.percentile is not None:
        click.echo(f"  Percentile: Top {result.percentile:.1f}% of HN stories")
    if result.expected_score is not None:
        click.echo(f"  Expected Score: ~{int(result.expected_score)} points (from multiclass model)")
    if result.percentile is not None or result.expected_score is not None:
        click.echo()

    # SHAP explanation (v2)
    if result.shap_features:
        click.echo("  --- Why This Score ---")
        for feat in result.shap_features:
            arrow = "↑" if feat["direction"] == "up" else "↓"
            click.echo(f"  {arrow} {feat['feature']} ({feat['importance']:+.2f})")
        click.echo()

    # Time recommendation (v2)
    if result.time_recommendation:
        click.echo("  --- Posting Advice ---")
        click.echo(f"  {result.time_recommendation}")
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
    from hackernews_simulator.config import PROCESSED_DIR
    score_model_path = MODELS_DIR / "score_model.txt"
    comment_model_path = MODELS_DIR / "comment_model.txt"
    multiclass_path = MODELS_DIR / "multiclass_model.txt"
    sorted_scores_path = PROCESSED_DIR / "sorted_scores.npy"
    time_stats_path = PROCESSED_DIR / "time_stats.json"

    try:
        simulator = HNSimulator(
            score_model_path=score_model_path,
            comment_model_path=comment_model_path,
            lancedb_path=LANCEDB_DIR,
            multiclass_model_path=multiclass_path if multiclass_path.exists() else None,
            sorted_scores_path=sorted_scores_path if sorted_scores_path.exists() else None,
            time_stats_path=time_stats_path if time_stats_path.exists() else None,
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
        from hackernews_simulator.rich_output import print_prediction
        print_prediction(title, result)


@main.command()
@click.option("--sample-size", default=100_000, show_default=True, help="Number of stories to fetch and train on")
@click.option("--min-score", default=1, show_default=True, help="Minimum story score to include")
def train(sample_size: int, min_score: int) -> None:
    """Run full training pipeline: fetch, preprocess, build features, train, save."""
    ensure_dirs()
    click.echo(f"Starting training pipeline (sample_size={sample_size}, min_score={min_score})...")

    click.echo("Step 1/4: Fetching data...")
    from hackernews_simulator.data.fetch import fetch_stories_stratified
    df = fetch_stories_stratified(total_limit=sample_size)
    stories_path = RAW_DIR / "stories.parquet"
    df.to_parquet(stories_path)
    click.echo(f"  Fetched {len(df)} stories -> {stories_path}")

    click.echo("Step 2/4: Preprocessing...")
    from hackernews_simulator.data.preprocess import preprocess_stories
    df = preprocess_stories(df)
    click.echo(f"  Preprocessed {len(df)} stories")

    click.echo("Step 3/4: Building feature matrix...")
    from hackernews_simulator.features.pipeline import build_feature_matrix
    from hackernews_simulator.config import PROCESSED_DIR
    import numpy as np
    X, feature_names = build_feature_matrix(df)
    np.save(PROCESSED_DIR / "features.npy", X)
    click.echo(f"  Features shape: {X.shape}")

    click.echo("Step 4/4: Training models...")
    from hackernews_simulator.model.train import train_model, save_model
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
    from hackernews_simulator.data.fetch import fetch_stories_stratified
    df = fetch_stories_stratified(total_limit=sample_size)
    path = out / "stories.parquet"
    df.to_parquet(path)
    click.echo(f"Saved {len(df)} stories to {path}")


@main.command()
@click.option("--features-dir", default=None, help="Directory with processed features (default: PROCESSED_DIR)")
def backtest(features_dir: str | None) -> None:
    """Run backtest on stored feature data and print report."""
    from hackernews_simulator.config import PROCESSED_DIR
    import json as _json
    from pathlib import Path as _Path

    src = _Path(features_dir) if features_dir else PROCESSED_DIR
    features_path = src / "features.npy"
    labels_path = src / "labels_score.npy"
    names_path = src / "feature_names.json"

    if not features_path.exists() or not labels_path.exists():
        click.echo(f"Features not found at {src}. Run 'train' first.", err=True)
        sys.exit(1)

    click.echo("Loading features...")
    X = np.load(str(features_path))
    y = np.load(str(labels_path))
    with open(str(names_path)) as f:
        feature_names = _json.load(f)

    click.echo(f"Running backtest on {len(X)} samples...")
    from hackernews_simulator.model.backtest import run_backtest, format_backtest_report
    results = run_backtest(X, y, feature_names)
    click.echo(format_backtest_report(results))


@main.command("suggest-loop")
@click.option("--title", required=True, help="Story title to optimize")
@click.option("--description", default="", help="Optional description")
@click.option("--max-iterations", default=5, show_default=True, help="Maximum optimization iterations")
def suggest_loop(title: str, description: str, max_iterations: int) -> None:
    """Iteratively optimize a title using the suggest loop."""
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

    click.echo(f"Optimizing: {title!r}")
    click.echo(f"Running up to {max_iterations} iteration(s)...")

    from hackernews_simulator.suggest import iterative_optimize
    result = iterative_optimize(
        simulator=simulator,
        original={"title": title, "description": description},
        client=None,
        max_iterations=max_iterations,
    )

    sep = "═" * 43
    click.echo(sep)
    click.echo("  Suggest Loop Results")
    click.echo(sep)
    click.echo(f"  Iterations run: {result['iterations']}")
    click.echo(f"  Score improvement: +{result['improvement']:.1f}")
    click.echo()
    click.echo(f"  Best title: {result['best']['title']}")
    click.echo(f"  Best score: ~{int(result['best']['predicted_score'])} points")
    if result["all_variants"]:
        click.echo()
        click.echo("  All variants tried:")
        for v in result["all_variants"]:
            click.echo(f"    ~{int(v['predicted_score']):<6} {v['title']}")
    click.echo(sep)


@main.command("compare")
@click.option("--file", "variants_file", required=True, help="YAML file with list of title variants")
def compare(variants_file: str) -> None:
    """Compare multiple title variants and print a rich table."""
    import yaml
    from pathlib import Path
    from hackernews_simulator.rich_output import print_comparison

    path = Path(variants_file)
    if not path.exists():
        click.echo(f"File not found: {variants_file}", err=True)
        sys.exit(1)

    with open(path) as f:
        variants = yaml.safe_load(f)

    if not isinstance(variants, list):
        click.echo("YAML file must contain a list of variant objects with at least a 'title' key.", err=True)
        sys.exit(1)

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

    results = []
    for v in variants:
        title = v.get("title", "")
        description = v.get("description", "")
        result = simulator.simulate(title=title, description=description, generate_comments=False)
        d = result.to_dict()
        d["title"] = title
        results.append(d)

    print_comparison(results)


@main.command()
@click.option("--from-scratch", is_flag=True, default=False,
              help="Fetch data and train from scratch instead of downloading pre-trained artifacts")
def init(from_scratch: bool) -> None:
    """Set up hackernews-simulator: download pre-trained models or train from scratch."""
    from hackernews_simulator.config import DATA_DIR, ensure_dirs
    ensure_dirs()

    if from_scratch:
        click.echo("Training from scratch...")
        click.echo("Step 1/4: Fetching data...")
        from hackernews_simulator.data.fetch import fetch_stories_stratified
        df = fetch_stories_stratified(total_limit=100_000)
        stories_path = RAW_DIR / "stories.parquet"
        df.to_parquet(stories_path)
        click.echo(f"  Fetched {len(df)} stories -> {stories_path}")

        click.echo("Step 2/4: Preprocessing...")
        from hackernews_simulator.data.preprocess import preprocess_stories
        df = preprocess_stories(df)
        click.echo(f"  Preprocessed {len(df)} stories")

        click.echo("Step 3/4: Building feature matrix and training models...")
        from hackernews_simulator.features.pipeline import build_feature_matrix
        from hackernews_simulator.config import PROCESSED_DIR
        X, feature_names = build_feature_matrix(df)
        np.save(PROCESSED_DIR / "features.npy", X)
        from hackernews_simulator.model.train import train_model, save_model
        score_labels = df["score"].values
        comment_labels = df["descendants"].fillna(0).values
        score_model = train_model(X, score_labels)
        comment_model = train_model(X, comment_labels)
        save_model(score_model, MODELS_DIR / "score_model.txt")
        save_model(comment_model, MODELS_DIR / "comment_model.txt")
        click.echo("  Models saved.")

        click.echo("Step 4/4: Building LanceDB index...")
        import pandas as pd
        from hackernews_simulator.rag.index import build_index as _build_index
        _build_index(df, LANCEDB_DIR)
        click.echo(f"  Index built at {LANCEDB_DIR}")

        click.echo("Done! Models trained and index built.")
    else:
        click.echo("Downloading pre-trained models from HuggingFace...")
        from hackernews_simulator.artifacts import check_artifacts, download_artifacts, download_lancedb
        from hackernews_simulator.config import LANCEDB_DIR

        if check_artifacts(DATA_DIR):
            click.echo("Models already downloaded. Use --from-scratch to retrain.")
            return

        download_artifacts(DATA_DIR)
        download_lancedb(LANCEDB_DIR)
        click.echo("Done! Run: hn-sim predict --title 'Your title here'")


@main.command()
@click.option("--port", default=8501, show_default=True, help="Port to run Streamlit on")
def ui(port: int) -> None:
    """Launch the Streamlit web UI."""
    import subprocess
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent.parent / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)])


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
    from hackernews_simulator.rag.index import build_index as _build_index
    df = pd.read_parquet(stories_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    _build_index(df, LANCEDB_DIR)
    click.echo(f"Index built at {LANCEDB_DIR}")
