"""Download pre-trained artifacts from HuggingFace Hub."""
from __future__ import annotations
from pathlib import Path

HF_REPO_ID = "malakhov/hackernews-simulator"

ARTIFACT_MANIFEST = {
    "score_model.txt": "models/score_model.txt",
    "comment_model.txt": "models/comment_model.txt",
    "multiclass_model.txt": "models/multiclass_model.txt",
    "sorted_scores.npy": "processed/sorted_scores.npy",
    "time_stats.json": "processed/time_stats.json",
    "domain_stats.json": "processed/domain_stats.json",
}

def check_artifacts(data_dir: Path) -> bool:
    """Check if all required artifacts exist."""
    for filename, rel_path in ARTIFACT_MANIFEST.items():
        if not (data_dir / rel_path).exists():
            return False
    return True

def download_artifacts(data_dir: Path) -> None:
    """Download model artifacts from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    for filename, rel_path in ARTIFACT_MANIFEST.items():
        target = data_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {filename}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=rel_path,
            local_dir=str(data_dir),
            repo_type="model",
        )

def download_lancedb(lancedb_dir: Path) -> None:
    """Download LanceDB index from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    lancedb_dir.parent.mkdir(parents=True, exist_ok=True)
    print("  Downloading LanceDB index...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=["lancedb/*"],
        local_dir=str(lancedb_dir.parent),
        repo_type="model",
    )
