"""Build LanceDB vector index from raw stories and comments parquet."""
import click
import pandas as pd

from hackernews_simulator.config import LANCEDB_DIR, RAW_DIR, ensure_dirs
from hackernews_simulator.data.preprocess import preprocess_comments, preprocess_stories
from hackernews_simulator.features.text import extract_title_embeddings
from hackernews_simulator.rag.index import build_comment_index, build_story_index


@click.command()
@click.option("--stories", "stories_path", default=None, help="Path to raw stories.parquet (default: RAW_DIR/stories.parquet)")
@click.option("--comments", "comments_path", default=None, help="Path to raw comments.parquet (default: RAW_DIR/comments.parquet)")
@click.option("--db", "db_path", default=None, help="LanceDB directory (default: LANCEDB_DIR)")
@click.option("--no-comments", "skip_comments", is_flag=True, default=False, help="Skip comment index")
def main(stories_path, comments_path, db_path, skip_comments):
    ensure_dirs()

    stories_src = stories_path or str(RAW_DIR / "stories.parquet")
    comments_src = comments_path or str(RAW_DIR / "comments.parquet")
    db = db_path or str(LANCEDB_DIR)

    click.echo(f"Loading stories from {stories_src}...")
    stories_df = pd.read_parquet(stories_src)
    click.echo(f"Loaded {len(stories_df)} stories")

    click.echo("Preprocessing stories...")
    stories_df = preprocess_stories(stories_df)

    click.echo("Generating story title embeddings...")
    embeddings = extract_title_embeddings(stories_df)

    click.echo(f"Building story index in {db}...")
    build_story_index(stories_df, embeddings, db_path=db)
    click.echo(f"Story index built: {len(stories_df)} rows, embedding dim {embeddings.shape[1]}")

    if not skip_comments:
        import os
        if not os.path.exists(comments_src):
            click.echo(f"No comments file found at {comments_src}, skipping comment index")
        else:
            click.echo(f"Loading comments from {comments_src}...")
            comments_df = pd.read_parquet(comments_src)
            click.echo(f"Loaded {len(comments_df)} comments")

            click.echo("Preprocessing comments...")
            comments_df = preprocess_comments(comments_df)
            click.echo(f"After preprocessing: {len(comments_df)} comments")

            click.echo(f"Building comment index in {db}...")
            build_comment_index(comments_df, db_path=db)
            click.echo(f"Comment index built: {len(comments_df)} rows")


if __name__ == "__main__":
    main()
