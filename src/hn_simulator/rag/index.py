"""LanceDB index building for stories and comments."""
from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np
import pandas as pd


def build_story_index(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    db_path: Path,
    table_name: str = "stories",
) -> None:
    """Create LanceDB table for stories with vector embeddings.

    Columns stored: id, title, url, clean_text, score, descendants, domain,
    post_type, vector.
    """
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(db_path)

    data = []
    for i, (_, row) in enumerate(df.iterrows()):
        data.append({
            "id": int(row["id"]),
            "title": str(row.get("title", "") or ""),
            "url": str(row.get("url", "") or ""),
            "clean_text": str(row.get("clean_text", "") or ""),
            "score": int(row.get("score", 0) or 0),
            "descendants": int(row.get("descendants", 0) or 0),
            "domain": str(row.get("domain", "") or ""),
            "post_type": str(row.get("post_type", "") or ""),
            "vector": embeddings[i].tolist(),
        })

    db.create_table(table_name, data, mode="overwrite")


def build_comment_index(
    df: pd.DataFrame,
    db_path: Path,
    table_name: str = "comments",
) -> None:
    """Create LanceDB table for comments (no vector, retrieved by parent ID).

    Columns stored: id, parent, clean_text, by.
    """
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(db_path)

    data = []
    for _, row in df.iterrows():
        data.append({
            "id": int(row["id"]),
            "parent": int(row["parent"]),
            "clean_text": str(row.get("clean_text", "") or ""),
            "by": str(row.get("by", "") or ""),
        })

    db.create_table(table_name, data, mode="overwrite")
