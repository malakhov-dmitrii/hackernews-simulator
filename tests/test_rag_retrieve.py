"""Tests for RAG retrieval.
Uses open-index schema: id (uint32), parent (uint32).
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def populated_lancedb(sample_stories_df, sample_comments_df, tmp_path):
    """Build a small LanceDB index for testing retrieval. Fixed seed."""
    from hn_simulator.rag.index import build_story_index, build_comment_index
    from hn_simulator.data.preprocess import preprocess_stories, preprocess_comments

    rng = np.random.default_rng(42)
    df = preprocess_stories(sample_stories_df)
    comments = preprocess_comments(sample_comments_df)
    n = len(df)
    embeddings = rng.standard_normal((n, 384)).astype(np.float32)
    db_path = tmp_path / "lancedb"
    build_story_index(df, embeddings, db_path=db_path)
    build_comment_index(comments, db_path=db_path)
    return db_path


class TestRetrieveSimilar:
    def test_returns_results(self, populated_lancedb):
        from hn_simulator.rag.retrieve import retrieve_similar_stories
        rng = np.random.default_rng(99)
        query_embedding = rng.standard_normal(384).astype(np.float32)
        results = retrieve_similar_stories(query_embedding, db_path=populated_lancedb, top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_results_have_required_fields(self, populated_lancedb):
        from hn_simulator.rag.retrieve import retrieve_similar_stories
        rng = np.random.default_rng(99)
        query_embedding = rng.standard_normal(384).astype(np.float32)
        results = retrieve_similar_stories(query_embedding, db_path=populated_lancedb, top_k=3)
        for r in results:
            assert "title" in r
            assert "score" in r
            assert "id" in r  # open-index column

    def test_retrieve_comments_for_story(self, populated_lancedb):
        from hn_simulator.rag.retrieve import retrieve_comments_for_story
        # Story id=1 has 2 comments in sample data (parent=1)
        comments = retrieve_comments_for_story(story_id=1, db_path=populated_lancedb, limit=10)
        assert isinstance(comments, list)
        assert len(comments) <= 10

    def test_retrieve_comments_where_filter_correctness(self, populated_lancedb):
        from hn_simulator.rag.retrieve import retrieve_comments_for_story
        # Story id=1 has exactly 2 comments (parent=[1,1,2,3] in fixture)
        comments_1 = retrieve_comments_for_story(story_id=1, db_path=populated_lancedb, limit=10)
        assert len(comments_1) == 2
        assert all(c["parent"] == 1 for c in comments_1)
        # Story id=2 has exactly 1 comment
        comments_2 = retrieve_comments_for_story(story_id=2, db_path=populated_lancedb, limit=10)
        assert len(comments_2) == 1
        assert comments_2[0]["parent"] == 2
        # Story id=99999 has no comments
        comments_none = retrieve_comments_for_story(story_id=99999, db_path=populated_lancedb, limit=10)
        assert comments_none == []


class TestRetrieveErrorPaths:
    def test_retrieve_from_nonexistent_table(self, tmp_path):
        """Error path: table doesn't exist."""
        from hn_simulator.rag.retrieve import retrieve_similar_stories
        import lancedb
        # Create empty db
        db_path = tmp_path / "empty_lancedb"
        lancedb.connect(db_path)
        rng = np.random.default_rng(99)
        query = rng.standard_normal(384).astype(np.float32)
        with pytest.raises((ValueError, FileNotFoundError)):
            retrieve_similar_stories(query, db_path=db_path, top_k=3)

    def test_retrieve_comments_empty_result(self, populated_lancedb):
        from hn_simulator.rag.retrieve import retrieve_comments_for_story
        # Story id=99999 has no comments
        comments = retrieve_comments_for_story(story_id=99999, db_path=populated_lancedb, limit=10)
        assert comments == []
