"""End-to-end integration test with synthetic data.

This test validates the full pipeline: preprocess -> features -> train -> predict -> simulate.
Uses synthetic/small data to avoid network calls and long runtimes.
Uses mock_embedding_model to avoid loading real SentenceTransformer.
All fixtures use open-index/hacker-news schema.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def full_pipeline(tmp_path, sample_stories_df, sample_comments_df, mock_embedding_model):
    """Build complete pipeline artifacts from sample data."""
    from hn_simulator.data.preprocess import preprocess_stories, preprocess_comments
    from hn_simulator.features.pipeline import build_feature_matrix
    from hn_simulator.model.train import train_score_model, train_comment_count_model, save_model
    from hn_simulator.rag.index import build_story_index, build_comment_index

    # Preprocess
    stories = preprocess_stories(sample_stories_df, min_score=0)
    comments = preprocess_comments(sample_comments_df)

    # Features — uses mock embedding model
    X, feature_names = build_feature_matrix(stories)
    y_score = stories["score"].values.astype(np.float32)
    y_comments = stories["descendants"].values.astype(np.float32)  # open-index: descendants

    # Train (use all data for both train/val since it's tiny)
    score_model, score_metrics = train_score_model(X, y_score, X, y_score, feature_names)
    comment_model, comment_metrics = train_comment_count_model(X, y_comments, X, y_comments, feature_names)

    # Save models
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    save_model(score_model, models_dir / "score_model.txt")
    save_model(comment_model, models_dir / "comment_model.txt")

    # Build RAG index — uses mock embeddings
    from hn_simulator.features.text import extract_title_embeddings
    embeddings = extract_title_embeddings(stories)
    lancedb_dir = tmp_path / "lancedb"
    build_story_index(stories, embeddings, db_path=lancedb_dir)
    build_comment_index(comments, db_path=lancedb_dir)

    return {
        "score_model_path": models_dir / "score_model.txt",
        "comment_model_path": models_dir / "comment_model.txt",
        "lancedb_path": lancedb_dir,
        "score_metrics": score_metrics,
        "comment_metrics": comment_metrics,
    }


class TestEndToEnd:
    def test_full_prediction_pipeline(self, full_pipeline, mock_embedding_model):
        from hn_simulator.simulator import HNSimulator

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""[
            {"username": "pg_fan", "comment": "This is exactly what the community needs.", "tone": "supportive"},
            {"username": "skeptic42", "comment": "How does this handle edge cases?", "tone": "skeptical"}
        ]""")]
        mock_client.messages.create.return_value = mock_response

        sim = HNSimulator(
            score_model_path=full_pipeline["score_model_path"],
            comment_model_path=full_pipeline["comment_model_path"],
            lancedb_path=full_pipeline["lancedb_path"],
            claude_client=mock_client,
        )

        result = sim.simulate(
            title="Show HN: I built an ML model to predict HN reactions",
            description="Uses LightGBM and RAG to simulate how HN would react to your project pitch.",
        )

        # Verify all result fields
        assert result.predicted_score >= 0
        assert result.predicted_comments >= 0
        assert result.reception_label in ("flop", "moderate", "hot", "viral")
        assert 0 < result.confidence <= 1.0
        assert sum(result.label_distribution.values()) == pytest.approx(1.0, abs=0.01)
        assert len(result.simulated_comments) == 2
        assert len(result.similar_stories) > 0

        # Verify dict serialization
        d = result.to_dict()
        assert isinstance(d, dict)
        assert all(key in d for key in [
            "predicted_score", "predicted_comments", "reception_label",
            "confidence", "simulated_comments", "similar_stories"
        ])

    def test_prediction_without_comments(self, full_pipeline, mock_embedding_model):
        from hn_simulator.simulator import HNSimulator

        sim = HNSimulator(
            score_model_path=full_pipeline["score_model_path"],
            comment_model_path=full_pipeline["comment_model_path"],
            lancedb_path=full_pipeline["lancedb_path"],
            claude_client=None,
        )

        result = sim.simulate(
            title="Ask HN: What are you working on this weekend?",
            generate_comments=False,
        )
        assert result.predicted_score >= 0
        assert result.simulated_comments == []
