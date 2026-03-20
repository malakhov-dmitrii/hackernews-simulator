"""Tests for AI-powered variant suggestion."""
import pytest
from unittest.mock import MagicMock
import json
import numpy as np


@pytest.fixture
def mock_models_and_simulator(tmp_path, mock_embedding_model):
    """Create a small trained HNSimulator for suggest tests."""
    from hackernews_simulator.model.train import train_score_model, train_comment_count_model, save_model
    from hackernews_simulator.rag.index import build_story_index, build_comment_index
    from hackernews_simulator.simulator import HNSimulator
    import pandas as pd

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 399)).astype(np.float32)
    y_score = (X[:, 0] * 10 + rng.standard_normal(100)).clip(0, None)
    y_comments = (X[:, 1] * 5 + rng.standard_normal(100)).clip(0, None)
    names = [f"feat_{i}" for i in range(399)]

    score_model, _ = train_score_model(X[:80], y_score[:80], X[80:], y_score[80:], names)
    comment_model, _ = train_comment_count_model(X[:80], y_comments[:80], X[80:], y_comments[80:], names)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    save_model(score_model, models_dir / "score_model.txt")
    save_model(comment_model, models_dir / "comment_model.txt")

    stories_df = pd.DataFrame({
        "id": np.array([1], dtype=np.uint32),
        "title": ["Test story"],
        "url": ["https://example.com"],
        "clean_text": ["Some text"],
        "score": np.array([50], dtype=np.int32),
        "descendants": np.array([10], dtype=np.int32),
        "domain": ["example.com"],
        "post_type": ["regular"],
    })
    embeddings = rng.standard_normal((1, 384)).astype(np.float32)
    lancedb_dir = tmp_path / "lancedb"
    build_story_index(stories_df, embeddings, db_path=lancedb_dir)

    comments_df = pd.DataFrame({
        "id": np.array([101], dtype=np.uint32),
        "parent": np.array([1], dtype=np.uint32),
        "clean_text": ["This is a test comment"],
        "by": ["testuser"],
    })
    build_comment_index(comments_df, db_path=lancedb_dir)

    sim = HNSimulator(
        score_model_path=models_dir / "score_model.txt",
        comment_model_path=models_dir / "comment_model.txt",
        lancedb_path=lancedb_dir,
        claude_client=None,
    )
    return sim


class TestSuggestVariants:
    def test_returns_list_of_variants(self):
        from hackernews_simulator.suggest import suggest_variants
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"title": "Show HN: Predict your HN score before posting", "description": "Open-source tool using 47M posts..."},
            {"title": "How we modeled HN culture with LightGBM and RAG", "description": "Technical deep-dive..."},
            {"title": "Ask HN: Is HN reaction predictable? We tested it", "description": "Our findings from analyzing..."},
        ]))]
        mock_client.messages.create.return_value = mock_response

        original = {"title": "My HN predictor", "description": "Predicts HN reactions"}
        suggestions = suggest_variants(original, client=mock_client, num_suggestions=3)
        assert isinstance(suggestions, list)
        assert len(suggestions) == 3
        for s in suggestions:
            assert "title" in s
            assert "description" in s

    def test_handles_malformed_response(self):
        from hackernews_simulator.suggest import suggest_variants
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not json")]
        mock_client.messages.create.return_value = mock_response

        original = {"title": "Test", "description": "Desc"}
        suggestions = suggest_variants(original, client=mock_client)
        assert suggestions == []

    def test_without_client_falls_back_to_cli(self):
        from hackernews_simulator.suggest import suggest_variants
        from unittest.mock import patch
        # Without client, should attempt Claude CLI (which we mock here)
        with patch("hackernews_simulator.claude_runner.run_claude", return_value='[{"title": "CLI suggestion", "description": "via CLI"}]'):
            suggestions = suggest_variants({"title": "T", "description": "D"}, client=None)
            assert len(suggestions) == 1
            assert suggestions[0]["title"] == "CLI suggestion"


class TestSuggestAndScore:
    def test_returns_scored_suggestions(self, mock_models_and_simulator):
        from hackernews_simulator.suggest import suggest_and_score
        sim = mock_models_and_simulator

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"title": "Suggested Title 1", "description": "Desc 1"},
            {"title": "Suggested Title 2", "description": "Desc 2"},
        ]))]
        mock_client.messages.create.return_value = mock_response

        original = {"title": "Original", "description": "Original desc"}
        results = suggest_and_score(sim, original, client=mock_client, num_suggestions=2)

        assert len(results) >= 2  # suggestions + original
        # Each has a score
        for r in results:
            assert r["predicted_score"] >= 0
            assert "title" in r
            assert "is_original" in r

    def test_includes_original_in_ranking(self, mock_models_and_simulator):
        from hackernews_simulator.suggest import suggest_and_score
        sim = mock_models_and_simulator

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"title": "Suggested", "description": "Desc"},
        ]))]
        mock_client.messages.create.return_value = mock_response

        original = {"title": "Original", "description": "Desc"}
        results = suggest_and_score(sim, original, client=mock_client, num_suggestions=1)

        originals = [r for r in results if r["is_original"]]
        assert len(originals) == 1


class TestIterativeOptimize:
    def _make_client(self, suggestions_per_call):
        """Return a mock client that cycles through suggestions_per_call lists."""
        mock_client = MagicMock()
        responses = []
        for suggestions in suggestions_per_call:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=json.dumps(suggestions))]
            responses.append(mock_response)
        mock_client.messages.create.side_effect = responses
        return mock_client

    def test_iterative_returns_dict_with_required_keys(self, mock_models_and_simulator):
        from hackernews_simulator.suggest import iterative_optimize
        sim = mock_models_and_simulator

        mock_client = self._make_client([
            [{"title": "Better Title A", "description": "Desc A"},
             {"title": "Better Title B", "description": "Desc B"},
             {"title": "Better Title C", "description": "Desc C"}],
        ])

        original = {"title": "My Original Post", "description": "Original desc"}
        result = iterative_optimize(sim, original, client=mock_client, max_iterations=1, num_suggestions=3)

        assert isinstance(result, dict)
        assert "best" in result
        assert "all_variants" in result
        assert "iterations" in result
        assert "improvement" in result
        assert "title" in result["best"]
        assert "description" in result["best"]
        assert "predicted_score" in result["best"]

    def test_iterative_max_iterations_respected(self, mock_models_and_simulator):
        from hackernews_simulator.suggest import iterative_optimize
        sim = mock_models_and_simulator

        # Provide enough unique suggestions for each iteration with distinct titles
        # Use very high scores by patching simulate — but easier: provide unique titles per iteration
        # and force improvement by making simulate return high scores.
        # Since we can't easily control scores, we just verify iterations <= max_iterations.
        suggestions_per_call = []
        for i in range(10):
            suggestions_per_call.append([
                {"title": f"Unique Title Iter{i} A", "description": "Desc"},
                {"title": f"Unique Title Iter{i} B", "description": "Desc"},
                {"title": f"Unique Title Iter{i} C", "description": "Desc"},
            ])

        mock_client = self._make_client(suggestions_per_call)

        original = {"title": "Original", "description": "Desc"}
        max_iter = 3
        result = iterative_optimize(
            sim, original, client=mock_client,
            max_iterations=max_iter, min_improvement=0.0, num_suggestions=3,
        )

        assert result["iterations"] <= max_iter

    def test_iterative_convergence_stops_early(self, mock_models_and_simulator):
        from hackernews_simulator.suggest import iterative_optimize
        sim = mock_models_and_simulator

        # All calls return the same title — after first iteration it's in seen_titles
        # so subsequent calls produce no new suggestions → stops early
        same_suggestions = [
            {"title": "Same Title Always", "description": "Desc"},
            {"title": "Same Title Always", "description": "Desc"},
            {"title": "Same Title Always", "description": "Desc"},
        ]
        mock_client = self._make_client([same_suggestions] * 5)

        original = {"title": "Original", "description": "Desc"}
        result = iterative_optimize(
            sim, original, client=mock_client,
            max_iterations=5, min_improvement=0.0, num_suggestions=3,
        )

        # Should stop after 2 iterations: first sees new title, second finds it already seen
        assert result["iterations"] < 5

    def test_iterative_tracks_seen_titles(self, mock_models_and_simulator):
        from hackernews_simulator.suggest import iterative_optimize
        sim = mock_models_and_simulator

        mock_client = self._make_client([
            [{"title": "Title Alpha", "description": "Desc A"},
             {"title": "Title Beta", "description": "Desc B"},
             {"title": "Title Gamma", "description": "Desc C"}],
        ])

        original = {"title": "Original Post", "description": "Desc"}
        result = iterative_optimize(
            sim, original, client=mock_client, max_iterations=1, num_suggestions=3,
        )

        titles = [v["title"] for v in result["all_variants"]]
        assert len(titles) == len(set(titles)), "Duplicate titles found in all_variants"
