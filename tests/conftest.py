"""Shared test fixtures — all using open-index/hacker-news schema.

Schema reference:
  id (uint32), type (int8: 1=story,2=comment), by (string),
  time (timestamp), text (string/HTML), score (int32), title (string),
  url (string), descendants (int32), parent (uint32), kids (list<uint32>),
  dead (uint8), deleted (uint8)
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_stories_df():
    """Minimal stories DataFrame matching open-index/hacker-news schema.
    type=1 for stories. Uses correct column names: id, descendants, time, by.
    """
    return pd.DataFrame({
        "id": np.array([1, 2, 3, 4, 5], dtype=np.uint32),
        "type": np.array([1, 1, 1, 1, 1], dtype=np.int8),
        "title": [
            "Show HN: I built a tool to predict HN reactions",
            "Why Rust is eating the world",
            "Ask HN: What are you working on?",
            "Google announces new AI model",
            "",
        ],
        "url": [
            "https://github.com/user/hn-predictor",
            "https://blog.example.com/rust-eating-world",
            "",
            "https://blog.google/ai-model",
            "https://example.com/empty-title",
        ],
        "text": [
            "<p>I built this tool using ML to predict...</p>",
            "",
            "I&#x27;m curious what side projects everyone is working on.",
            "",
            "<p>Some <b>HTML</b> content &amp; entities</p>",
        ],
        "score": np.array([150, 45, 300, 80, 2], dtype=np.int32),
        "descendants": np.array([95, 23, 200, 40, 0], dtype=np.int32),
        "time": pd.to_datetime([
            "2022-06-15 14:30:00",
            "2022-09-01 09:00:00",
            "2023-03-10 16:00:00",
            "2023-07-20 11:00:00",
            "2024-01-05 08:00:00",
        ], utc=True),
        "by": ["user1", "user2", "user3", "user4", "user5"],
        "dead": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
        "deleted": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
    })


@pytest.fixture
def sample_comments_df():
    """Minimal comments DataFrame matching open-index/hacker-news schema.
    type=2 for comments. parent is uint32 referencing story id.
    """
    return pd.DataFrame({
        "id": np.array([101, 102, 103, 104], dtype=np.uint32),
        "type": np.array([2, 2, 2, 2], dtype=np.int8),
        "parent": np.array([1, 1, 2, 3], dtype=np.uint32),
        "text": [
            "<p>This is really cool! I&#x27;ve been wanting something like this.</p>",
            "<p>Have you considered using transformers instead of LightGBM?</p>",
            "Rust is great but the learning curve is steep.",
            "<p>Working on a <a href=\"https://example.com\">side project</a> in Go.</p>",
        ],
        "by": ["commenter1", "commenter2", "commenter3", "commenter4"],
        "time": pd.to_datetime([
            "2022-06-15 15:00:00",
            "2022-06-15 15:30:00",
            "2022-09-01 10:00:00",
            "2023-03-10 17:00:00",
        ], utc=True),
        "dead": np.array([0, 0, 0, 0], dtype=np.uint8),
        "deleted": np.array([0, 0, 0, 0], dtype=np.uint8),
    })


@pytest.fixture
def tmp_data_dirs(tmp_path):
    """Create temporary data directory structure."""
    dirs = {
        "data": tmp_path / "data",
        "raw": tmp_path / "data" / "raw",
        "processed": tmp_path / "data" / "processed",
        "models": tmp_path / "data" / "models",
        "lancedb": tmp_path / "data" / "lancedb",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    return dirs


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer.encode() to return deterministic 384-dim vectors.
    Use this for fast unit tests. Only @pytest.mark.slow tests load real model.
    """
    def deterministic_encode(texts, batch_size=64, show_progress_bar=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            text_seed = hash(text) % (2**31)
            text_rng = np.random.default_rng(text_seed)
            embeddings[i] = text_rng.standard_normal(384).astype(np.float32)
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings

    mock_model = MagicMock()
    mock_model.encode = deterministic_encode
    mock_model.get_sentence_embedding_dimension.return_value = 384

    with patch("hackernews_simulator.features.text.SentenceTransformer", return_value=mock_model):
        with patch("hackernews_simulator.features.text._model", None):
            yield mock_model
