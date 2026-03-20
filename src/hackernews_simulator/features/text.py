"""Text embedding features using sentence-transformers MiniLM-L6-v2."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# Suppress noisy model loading logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — patched by mock_embedding_model fixture in tests
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy singleton loader for all-MiniLM-L6-v2."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts using the singleton model.

    Returns float32 ndarray of shape (n, 384).
    """
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def extract_title_embeddings(df, batch_size: int = 64) -> np.ndarray:
    """Extract the title column from df and embed all titles.

    Returns float32 ndarray of shape (n, 384).
    """
    titles = df["title"].tolist()
    return embed_texts(titles, batch_size=batch_size)


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save embeddings array to a .npy file."""
    np.save(str(path), embeddings)


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load embeddings array from a .npy file.

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return np.load(str(path))
