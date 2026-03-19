"""Tests for LanceDB index building.
Uses open-index schema: id (uint32), parent (uint32), etc.
All random data uses fixed seed.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def stories_with_embeddings(sample_stories_df):
    """Stories DataFrame with pre-computed embeddings (fixed seed)."""
    from hn_simulator.data.preprocess import preprocess_stories
    rng = np.random.default_rng(42)
    df = preprocess_stories(sample_stories_df)
    n = len(df)
    embeddings = rng.standard_normal((n, 384)).astype(np.float32)
    return df, embeddings


class TestBuildStoryIndex:
    def test_creates_lance_table(self, stories_with_embeddings, tmp_path):
        from hn_simulator.rag.index import build_story_index
        df, embeddings = stories_with_embeddings
        db_path = tmp_path / "lancedb"
        build_story_index(df, embeddings, db_path=db_path)
        import lancedb
        db = lancedb.connect(db_path)
        assert "stories" in db.table_names()

    def test_table_has_correct_schema(self, stories_with_embeddings, tmp_path):
        from hn_simulator.rag.index import build_story_index
        df, embeddings = stories_with_embeddings
        db_path = tmp_path / "lancedb"
        build_story_index(df, embeddings, db_path=db_path)
        import lancedb
        db = lancedb.connect(db_path)
        table = db.open_table("stories")
        schema_names = [f.name for f in table.schema]
        assert "title" in schema_names
        assert "vector" in schema_names
        assert "score" in schema_names
        assert "id" in schema_names  # open-index column

    def test_table_has_correct_row_count(self, stories_with_embeddings, tmp_path):
        from hn_simulator.rag.index import build_story_index
        df, embeddings = stories_with_embeddings
        db_path = tmp_path / "lancedb"
        build_story_index(df, embeddings, db_path=db_path)
        import lancedb
        db = lancedb.connect(db_path)
        table = db.open_table("stories")
        assert len(table) == len(df)


class TestBuildCommentIndex:
    def test_creates_comment_table(self, sample_comments_df, tmp_path):
        from hn_simulator.rag.index import build_comment_index
        from hn_simulator.data.preprocess import preprocess_comments
        comments = preprocess_comments(sample_comments_df)
        db_path = tmp_path / "lancedb"
        build_comment_index(comments, db_path=db_path)
        import lancedb
        db = lancedb.connect(db_path)
        assert "comments" in db.table_names()

    def test_comment_table_has_parent_column(self, sample_comments_df, tmp_path):
        from hn_simulator.rag.index import build_comment_index
        from hn_simulator.data.preprocess import preprocess_comments
        comments = preprocess_comments(sample_comments_df)
        db_path = tmp_path / "lancedb"
        build_comment_index(comments, db_path=db_path)
        import lancedb
        db = lancedb.connect(db_path)
        table = db.open_table("comments")
        schema_names = [f.name for f in table.schema]
        assert "parent" in schema_names
