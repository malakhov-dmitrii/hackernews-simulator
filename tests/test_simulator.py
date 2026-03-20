"""Tests for the simulator orchestrator.
All random data uses fixed seed. Uses mock embedding model.
Uses open-index schema: id (uint32), descendants, time, by.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture
def mock_models(tmp_path, mock_embedding_model):
    """Create mock model files and LanceDB index. Fixed seed."""
    from hn_simulator.model.train import train_score_model, train_comment_count_model, save_model
    from hn_simulator.rag.index import build_story_index, build_comment_index
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

    # Small LanceDB index — open-index schema
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

    return {
        "score_model_path": models_dir / "score_model.txt",
        "comment_model_path": models_dir / "comment_model.txt",
        "lancedb_path": lancedb_dir,
    }


class TestSimulator:
    def test_simulate_returns_result(self, mock_models, mock_embedding_model):
        from hn_simulator.simulator import HNSimulator

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"username": "user1", "comment": "Cool!", "tone": "positive"}]')]
        mock_client.messages.create.return_value = mock_response

        sim = HNSimulator(
            score_model_path=mock_models["score_model_path"],
            comment_model_path=mock_models["comment_model_path"],
            lancedb_path=mock_models["lancedb_path"],
            claude_client=mock_client,
        )
        result = sim.simulate("Show HN: Test Project", "A test project description")
        assert result.predicted_score >= 0
        assert result.predicted_comments >= 0
        assert result.reception_label in ("flop", "low", "moderate", "hot", "viral")
        assert isinstance(result.simulated_comments, list)
        assert isinstance(result.similar_stories, list)

    def test_simulate_without_comments(self, mock_models, mock_embedding_model):
        from hn_simulator.simulator import HNSimulator

        sim = HNSimulator(
            score_model_path=mock_models["score_model_path"],
            comment_model_path=mock_models["comment_model_path"],
            lancedb_path=mock_models["lancedb_path"],
            claude_client=None,  # No API client
        )
        result = sim.simulate("Test", "Desc", generate_comments=False)
        assert result.predicted_score >= 0
        assert result.simulated_comments == []


class TestSimulationResult:
    def test_to_dict(self):
        from hn_simulator.simulator import SimulationResult
        result = SimulationResult(
            predicted_score=42.5,
            predicted_comments=15.0,
            reception_label="hot",
            confidence=0.75,
            label_distribution={"flop": 0.05, "moderate": 0.15, "hot": 0.55, "viral": 0.25},
            simulated_comments=[{"username": "u1", "comment": "Nice!", "tone": "positive"}],
            similar_stories=[{"title": "Similar", "score": 100}],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["predicted_score"] == 42.5
        assert d["reception_label"] == "hot"
