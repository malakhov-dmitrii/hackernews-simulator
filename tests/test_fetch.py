"""Tests for data fetching from open-index/hacker-news.
Query construction tests run without network.
Slow tests hit HuggingFace.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestQueryConstruction:
    def test_stories_query_selects_correct_columns(self):
        from hn_simulator.data.fetch import build_stories_query
        query = build_stories_query(limit=100)
        # Must use open-index column names
        assert "id" in query
        assert "title" in query
        assert "url" in query
        assert "score" in query
        assert '"by"' in query  # MUST be quoted — DuckDB reserved keyword
        assert "time" in query
        assert "descendants" in query
        # Must NOT use old column names
        assert "objectID" not in query
        assert "num_comments" not in query
        assert "created_at" not in query

    def test_stories_query_filters_type_1(self):
        from hn_simulator.data.fetch import build_stories_query
        query = build_stories_query(limit=100)
        assert "type = 1" in query or "type=1" in query

    def test_stories_query_filters_dead_deleted(self):
        from hn_simulator.data.fetch import build_stories_query
        query = build_stories_query(limit=100)
        assert "dead" in query.lower()  # dead = 0
        assert "deleted" in query.lower()  # deleted = 0

    def test_stories_query_filters_min_date(self):
        from hn_simulator.data.fetch import build_stories_query
        query = build_stories_query(limit=100, min_date="2020-01-01")
        assert "2020-01-01" in query

    def test_stories_query_applies_limit(self):
        from hn_simulator.data.fetch import build_stories_query
        query = build_stories_query(limit=500)
        assert "500" in query

    def test_comments_query_uses_correct_columns(self):
        from hn_simulator.data.fetch import build_comments_query
        query = build_comments_query(story_ids=[1, 2, 3])
        assert "parent" in query
        assert "text" in query
        assert '"by"' in query  # quoted
        assert "type = 2" in query or "type=2" in query

    def test_comments_query_filters_dead_deleted(self):
        from hn_simulator.data.fetch import build_comments_query
        query = build_comments_query(story_ids=[1])
        assert "dead" in query.lower()
        assert "deleted" in query.lower()

    def test_stratified_sampling_query(self):
        from hn_simulator.data.fetch import build_stratified_stories_query
        query = build_stratified_stories_query(total_limit=10000)
        # Should have UNION ALL for different score ranges
        assert "UNION ALL" in query or "union all" in query.lower()
        # Must filter type=1, dead=0, deleted=0
        assert "type = 1" in query or "type=1" in query

    def test_query_uses_hf_dataset_url(self):
        from hn_simulator.data.fetch import build_stories_query
        from hn_simulator.config import HF_DATASET_URL
        query = build_stories_query(limit=10)
        assert "open-index/hacker-news" in query


class TestDuckDBConnectionFailure:
    """Error path: DuckDB connection/query failure."""

    def test_fetch_raises_on_connection_error(self):
        from hn_simulator.data.fetch import fetch_stories
        with patch("hn_simulator.data.fetch.duckdb") as mock_duckdb:
            mock_duckdb.connect.side_effect = RuntimeError("Connection failed")
            with pytest.raises(RuntimeError, match="Connection failed"):
                fetch_stories(limit=10)


@pytest.mark.slow
class TestSmallFetch:
    """These tests hit HuggingFace — run with pytest -m slow."""

    def test_fetch_small_stories_sample(self):
        from hn_simulator.data.fetch import fetch_stories
        df = fetch_stories(limit=10, min_score=100, min_date="2023-01-01")
        assert len(df) <= 10
        assert "title" in df.columns
        assert "score" in df.columns
        assert "descendants" in df.columns
        assert all(df["score"] >= 100)
