"""Central configuration for HN Reaction Simulator."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LANCEDB_DIR = DATA_DIR / "lancedb"

# HuggingFace dataset — open-index/hacker-news
# Nested path: data/YYYY/YYYY-MM.parquet
# All item types in one dataset: type=1 (story), type=2 (comment), etc.
HF_DATASET_URL = "hf://datasets/open-index/hacker-news/data/*/*.parquet"

# For practical fetching, use specific years to avoid OOM on full glob
HF_DATASET_YEARS = [
    "hf://datasets/open-index/hacker-news/data/2024/*.parquet",
    "hf://datasets/open-index/hacker-news/data/2025/*.parquet",
]

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LightGBM parameters
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}

# Score distribution thresholds (calibrated from dataset stats)
SCORE_THRESHOLDS = {
    "flop": 3,
    "moderate": 15,
    "hot": 100,
    "viral": 300,
}

# RAG settings
RAG_TOP_K = 5
RAG_SAMPLE_SIZE = 300_000

# Temporal split
TRAIN_CUTOFF_DATE = "2023-01-01"
VAL_START_DATE = "2023-01-01"

# Comment generation
MAX_COMMENTS_TO_GENERATE = 5
CLAUDE_MODEL = "claude-sonnet-4-6-20250410"


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LANCEDB_DIR]:
        d.mkdir(parents=True, exist_ok=True)
