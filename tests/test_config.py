"""Tests for configuration module."""
import pytest
from pathlib import Path


def test_project_root_exists():
    from hn_simulator.config import PROJECT_ROOT
    assert isinstance(PROJECT_ROOT, Path)


def test_data_dirs_defined():
    from hn_simulator.config import DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LANCEDB_DIR
    assert all(isinstance(d, Path) for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LANCEDB_DIR])
    assert str(RAW_DIR).startswith(str(DATA_DIR))
    assert str(PROCESSED_DIR).startswith(str(DATA_DIR))
    assert str(MODELS_DIR).startswith(str(DATA_DIR))
    assert str(LANCEDB_DIR).startswith(str(DATA_DIR))


def test_hf_dataset_url():
    from hn_simulator.config import HF_DATASET_URL
    assert HF_DATASET_URL == "hf://datasets/open-index/hacker-news/data/*/*.parquet"


def test_model_params():
    from hn_simulator.config import LIGHTGBM_PARAMS
    assert isinstance(LIGHTGBM_PARAMS, dict)
    assert "objective" in LIGHTGBM_PARAMS
    assert "num_leaves" in LIGHTGBM_PARAMS


def test_score_thresholds():
    from hn_simulator.config import SCORE_THRESHOLDS
    assert "flop" in SCORE_THRESHOLDS
    assert "moderate" in SCORE_THRESHOLDS
    assert "hot" in SCORE_THRESHOLDS
    assert "viral" in SCORE_THRESHOLDS
    vals = [SCORE_THRESHOLDS[k] for k in ["flop", "moderate", "hot", "viral"]]
    assert vals == sorted(vals)


def test_embedding_model_name():
    from hn_simulator.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"


def test_rag_config():
    from hn_simulator.config import RAG_TOP_K, RAG_SAMPLE_SIZE
    assert isinstance(RAG_TOP_K, int) and RAG_TOP_K > 0
    assert isinstance(RAG_SAMPLE_SIZE, int) and RAG_SAMPLE_SIZE > 0


def test_temporal_split_date():
    from hn_simulator.config import TRAIN_CUTOFF_DATE, VAL_START_DATE
    assert TRAIN_CUTOFF_DATE == "2023-01-01"
    assert VAL_START_DATE == "2023-01-01"


def test_ensure_dirs_creates_directories(tmp_path, monkeypatch):
    from hn_simulator import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "data" / "models")
    monkeypatch.setattr(config, "LANCEDB_DIR", tmp_path / "data" / "lancedb")
    config.ensure_dirs()
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
    assert (tmp_path / "data" / "models").is_dir()
    assert (tmp_path / "data" / "lancedb").is_dir()
