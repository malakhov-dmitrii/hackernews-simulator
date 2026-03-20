"""LanceDB retrieval for stories and comments."""
from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np


def _open_db(db_path: Path) -> lancedb.DBConnection:
    """Open a LanceDB connection at db_path."""
    return lancedb.connect(Path(db_path))


def _table_names(db: lancedb.DBConnection) -> list[str]:
    """Return list of table names, compatible with LanceDB 0.30+."""
    return db.list_tables().tables


def retrieve_similar_stories(
    query_embedding: np.ndarray,
    db_path: Path,
    top_k: int = 5,
) -> list[dict]:
    """Return top_k stories most similar to query_embedding.

    Raises ValueError if the stories table does not exist.
    """
    db = _open_db(db_path)
    if "stories" not in _table_names(db):
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
    Uses pushed-down .where() filter for performance (~54x speedup vs full table scan).
    """
    db = _open_db(db_path)
    if "comments" not in _table_names(db):
        return []
    table = db.open_table("comments")
    try:
        results = table.search().where(f"parent = {story_id}").limit(limit).to_list()
    except Exception:
        # Fallback for older LanceDB versions
        df = table.to_pandas()
        filtered = df[df["parent"] == int(story_id)]
        results = filtered.head(limit).to_dict("records")
    return results
