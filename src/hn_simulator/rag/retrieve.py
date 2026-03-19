"""LanceDB retrieval for stories and comments."""
from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np


def _open_db(db_path: Path) -> lancedb.DBConnection:
    """Open a LanceDB connection at db_path."""
    return lancedb.connect(Path(db_path))


def retrieve_similar_stories(
    query_embedding: np.ndarray,
    db_path: Path,
    top_k: int = 5,
) -> list[dict]:
    """Return top_k stories most similar to query_embedding.

    Raises ValueError if the stories table does not exist.
    """
    db = _open_db(db_path)
    if "stories" not in db.table_names():
        raise ValueError(f"Table 'stories' not found in LanceDB at {db_path}")
    table = db.open_table("stories")
    results = table.search(query_embedding).limit(top_k).to_list()
    return results


def retrieve_comments_for_story(
    story_id: int,
    db_path: Path,
    limit: int = 20,
) -> list[dict]:
    """Return up to limit comments whose parent == story_id.

    Returns empty list if the comments table doesn't exist or no matches found.
    """
    db = _open_db(db_path)
    if "comments" not in db.table_names():
        return []
    table = db.open_table("comments")
    # Filter by numeric parent column using pandas
    df = table.to_pandas()
    filtered = df[df["parent"] == story_id]
    if limit:
        filtered = filtered.head(limit)
    return filtered.to_dict(orient="records")
