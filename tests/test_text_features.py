"""Tests for text embedding features.
Unit tests use mock_embedding_model fixture (no real model download).
Only @pytest.mark.slow tests load the real SentenceTransformer.
"""
import pytest
import numpy as np
import pandas as pd


class TestEmbeddingGenerationMocked:
    """Fast tests using mock embedding model."""

    def test_returns_numpy_array(self, mock_embedding_model):
        from hackernews_simulator.features.text import embed_texts
        texts = ["Hello world", "Another text"]
        result = embed_texts(texts)
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self, mock_embedding_model):
        from hackernews_simulator.features.text import embed_texts
        texts = ["Hello world", "Another text", "Third one"]
        result = embed_texts(texts)
        assert result.shape == (3, 384)

    def test_different_texts_different_embeddings(self, mock_embedding_model):
        from hackernews_simulator.features.text import embed_texts
        texts = ["Python is great", "Rust is fast"]
        result = embed_texts(texts)
        # Different text should produce different embeddings
        assert not np.allclose(result[0], result[1])

    def test_same_text_same_embedding(self, mock_embedding_model):
        from hackernews_simulator.features.text import embed_texts
        texts = ["Same text", "Same text"]
        result = embed_texts(texts)
        np.testing.assert_array_equal(result[0], result[1])

    def test_empty_text_handled(self, mock_embedding_model):
        from hackernews_simulator.features.text import embed_texts
        texts = ["Real text", ""]
        result = embed_texts(texts)
        assert result.shape == (2, 384)

    def test_batch_processing(self, mock_embedding_model):
        from hackernews_simulator.features.text import embed_texts
        texts = [f"Text number {i}" for i in range(100)]
        result = embed_texts(texts, batch_size=32)
        assert result.shape == (100, 384)


class TestTitleEmbeddingsMocked:
    def test_extract_from_dataframe(self, mock_embedding_model):
        from hackernews_simulator.features.text import extract_title_embeddings
        df = pd.DataFrame({"title": ["Show HN: My project", "Why Rust is great"]})
        result = extract_title_embeddings(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)


class TestEmbeddingModelLoadFailure:
    """Error path: model fails to load."""

    def test_raises_on_model_load_failure(self):
        from unittest.mock import patch
        with patch("hackernews_simulator.features.text.SentenceTransformer", side_effect=OSError("Model not found")):
            with patch("hackernews_simulator.features.text._model", None):
                from hackernews_simulator.features.text import embed_texts
                with pytest.raises(OSError, match="Model not found"):
                    embed_texts(["test"])


class TestSaveLoadEmbeddings:
    def test_save_and_load_roundtrip(self, tmp_path):
        from hackernews_simulator.features.text import save_embeddings, load_embeddings
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((10, 384)).astype(np.float32)
        path = tmp_path / "embeddings.npy"
        save_embeddings(arr, path)
        loaded = load_embeddings(path)
        np.testing.assert_array_equal(arr, loaded)

    def test_load_nonexistent_raises(self, tmp_path):
        from hackernews_simulator.features.text import load_embeddings
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent.npy")


@pytest.mark.slow
class TestEmbeddingGenerationReal:
    """Slow tests that load the actual SentenceTransformer model."""

    def test_real_model_produces_384_dim(self):
        from hackernews_simulator.features.text import embed_texts
        result = embed_texts(["Hello world"])
        assert result.shape == (1, 384)

    def test_real_model_semantically_similar(self):
        from hackernews_simulator.features.text import embed_texts
        texts = ["The cat sat on the mat", "A feline rested on the rug", "Python web framework"]
        result = embed_texts(texts)
        # First two should be more similar to each other than to third
        sim_01 = np.dot(result[0], result[1])
        sim_02 = np.dot(result[0], result[2])
        assert sim_01 > sim_02
