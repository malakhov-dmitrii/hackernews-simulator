"""Fetch stratified HN stories sample from HuggingFace via open-index/hacker-news."""
import click
from hackernews_simulator.config import RAW_DIR, ensure_dirs
from hackernews_simulator.data.fetch import fetch_stories_stratified, fetch_comments_for_stories


@click.command()
@click.option("--stories", default=100000, help="Number of stories to fetch")
@click.option("--comments/--no-comments", default=True, help="Also fetch comments")
@click.option("--seed", default=42, help="Random seed for reproducible sampling")
def main(stories, comments, seed):
    ensure_dirs()
    click.echo(f"Fetching {stories} stratified stories from open-index/hacker-news...")
    df = fetch_stories_stratified(total_limit=stories, seed=seed)
    path = RAW_DIR / "stories.parquet"
    df.to_parquet(path)
    click.echo(f"Saved {len(df)} stories to {path}")

    if comments:
        story_ids = df["id"].tolist()  # open-index: id is uint32
        click.echo(f"Fetching comments for {len(story_ids)} stories...")
        comments_df = fetch_comments_for_stories(story_ids)
        comments_path = RAW_DIR / "comments.parquet"
        comments_df.to_parquet(comments_path)
        click.echo(f"Saved {len(comments_df)} comments to {comments_path}")


if __name__ == "__main__":
    main()
