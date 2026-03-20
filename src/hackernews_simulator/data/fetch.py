"""Data fetching from HuggingFace open-index/hacker-news dataset via DuckDB."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import duckdb
import pandas as pd

from hackernews_simulator.config import HF_DATASET_URL, HF_DATASET_YEARS


def build_stories_query(
    limit: int,
    min_score: int = 0,
    min_date: str | None = None,
    seed: int | None = None,
) -> str:
    """Build SQL query to fetch stories from HuggingFace dataset.

    Selects open-index/hacker-news columns: id, title, url, text, score,
    descendants, "by" (quoted — DuckDB reserved word), time, dead, deleted.
    Filters: type = 1 (stories), dead = 0, deleted = 0, title IS NOT NULL.
    """
    where_clauses = [
        "type = 1",
        "dead = 0",
        "deleted = 0",
        "title IS NOT NULL",
        "title != ''",
    ]
    if min_score > 0:
        where_clauses.append(f"score >= {min_score}")
    if min_date is not None:
        where_clauses.append(f"time >= '{min_date}'")

    where_str = " AND ".join(where_clauses)

    if seed is not None:
        order_by = f"hash(id + {seed})"
    else:
        order_by = "score DESC"

    return f"""
SELECT
    id,
    title,
    url,
    text,
    score,
    descendants,
    "by",
    time,
    dead,
    deleted
FROM '{HF_DATASET_URL}'
WHERE {where_str}
ORDER BY {order_by}
LIMIT {limit}
""".strip()


def build_comments_query(story_ids: list[int]) -> str:
    """Build SQL query to fetch comments for given story IDs.

    Filters: type = 2 (comments), dead = 0, deleted = 0,
    parent IN (story_ids).
    """
    ids_str = ", ".join(str(i) for i in story_ids)
    return f"""
SELECT
    id,
    parent,
    text,
    "by",
    time,
    dead,
    deleted
FROM '{HF_DATASET_URL}'
WHERE type = 2
  AND dead = 0
  AND deleted = 0
  AND parent IN ({ids_str})
""".strip()


def build_stratified_stories_query(
    total_limit: int,
    seed: int | None = None,
) -> str:
    """Build UNION ALL query with stratified score-range sampling.

    Score buckets and proportions:
      [1, 5)   -> 30% of total_limit
      [5, 20)  -> 30% of total_limit
      [20, 100)-> 25% of total_limit
      [100+]   -> 15% of total_limit
    """
    buckets = [
        ("score >= 1 AND score < 5",   0.30),
        ("score >= 5 AND score < 20",  0.30),
        ("score >= 20 AND score < 100", 0.25),
        ("score >= 100",               0.15),
    ]

    base_where = (
        "type = 1 AND dead = 0 AND deleted = 0 "
        "AND title IS NOT NULL AND title != '' "
        "AND time >= '2018-01-01'"
    )

    if seed is not None:
        order_expr = f"hash(id + {seed})"
    else:
        order_expr = "random()"

    # Build a read_parquet call over specific year files to avoid OOM on full glob
    year_urls = ", ".join(f"'{u}'" for u in HF_DATASET_YEARS)
    source = f"read_parquet([{year_urls}])"

    parts = []
    for score_filter, proportion in buckets:
        bucket_limit = max(1, int(total_limit * proportion))
        parts.append(f"""(SELECT
    id,
    title,
    url,
    text,
    score,
    descendants,
    "by",
    time,
    dead,
    deleted
FROM {source}
WHERE {base_where}
  AND {score_filter}
ORDER BY {order_expr}
LIMIT {bucket_limit})""")

    return "\nUNION ALL\n".join(parts).strip()


@contextmanager
def _duckdb_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager for a DuckDB connection with httpfs loaded."""
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        yield con
    finally:
        con.close()


def fetch_stories(
    limit: int,
    min_score: int = 0,
    min_date: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fetch stories from HuggingFace via DuckDB.

    Returns a DataFrame with open-index/hacker-news story columns.
    """
    query = build_stories_query(limit=limit, min_score=min_score, min_date=min_date, seed=seed)
    with _duckdb_connection() as con:
        return con.execute(query).fetchdf()


def fetch_stories_stratified(
    total_limit: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fetch stories with stratified score-range sampling via DuckDB."""
    query = build_stratified_stories_query(total_limit=total_limit, seed=seed)
    with _duckdb_connection() as con:
        return con.execute(query).fetchdf()


def fetch_comments_for_stories(
    story_ids: list[int],
    limit_per_story: int = 50,
) -> pd.DataFrame:
    """Fetch comments for given story IDs via DuckDB.

    If len(story_ids) > 1000, batches in groups of 1000 and concatenates.
    """
    if not story_ids:
        return pd.DataFrame(columns=["id", "parent", "text", "by", "time", "dead", "deleted"])

    batch_size = 1000
    if len(story_ids) <= batch_size:
        query = build_comments_query(story_ids)
        with _duckdb_connection() as con:
            return con.execute(query).fetchdf()

    # Batch processing for large story_ids lists
    frames = []
    for i in range(0, len(story_ids), batch_size):
        batch = story_ids[i : i + batch_size]
        query = build_comments_query(batch)
        with _duckdb_connection() as con:
            frames.append(con.execute(query).fetchdf())

    return pd.concat(frames, ignore_index=True)
