"""Tests for artifact download helpers."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hackernews_simulator.artifacts import (
    ARTIFACT_MANIFEST,
    HF_REPO_ID,
    check_artifacts,
    download_artifacts,
    download_lancedb,
)


class TestArtifactManifest:
    def test_manifest_has_required_files(self):
        required = {
            "score_model.txt",
            "comment_model.txt",
            "multiclass_model.txt",
            "sorted_scores.npy",
            "time_stats.json",
            "domain_stats.json",
        }
        assert required == set(ARTIFACT_MANIFEST.keys())

    def test_manifest_values_are_relative_paths(self):
        for filename, rel_path in ARTIFACT_MANIFEST.items():
            assert "/" in rel_path, f"{filename} should have a subdirectory in its path"

    def test_hf_repo_id_is_set(self):
        assert HF_REPO_ID == "malakhov/hackernews-simulator"


class TestCheckArtifacts:
    def test_returns_false_when_files_missing(self, tmp_path):
        assert check_artifacts(tmp_path) is False

    def test_returns_false_when_some_files_missing(self, tmp_path):
        # Create only the first artifact
        first_rel = next(iter(ARTIFACT_MANIFEST.values()))
        target = tmp_path / first_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        assert check_artifacts(tmp_path) is False

    def test_returns_true_when_all_present(self, tmp_path):
        for filename, rel_path in ARTIFACT_MANIFEST.items():
            target = tmp_path / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        assert check_artifacts(tmp_path) is True


class TestDownloadArtifacts:
    def test_calls_hf_hub_download_for_each_artifact(self, tmp_path):
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            download_artifacts(tmp_path)
        assert mock_download.call_count == len(ARTIFACT_MANIFEST)

    def test_passes_correct_repo_id(self, tmp_path):
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            download_artifacts(tmp_path)
        for call in mock_download.call_args_list:
            assert call.kwargs["repo_id"] == HF_REPO_ID

    def test_creates_parent_dirs(self, tmp_path):
        with patch("huggingface_hub.hf_hub_download"):
            download_artifacts(tmp_path)
        # Parent directories should be created before download
        for filename, rel_path in ARTIFACT_MANIFEST.items():
            parent = (tmp_path / rel_path).parent
            assert parent.exists()


class TestDownloadLancedb:
    def test_calls_snapshot_download(self, tmp_path):
        lancedb_dir = tmp_path / "lancedb"
        with patch("huggingface_hub.snapshot_download") as mock_snap:
            download_lancedb(lancedb_dir)
        mock_snap.assert_called_once()
        call_kwargs = mock_snap.call_args.kwargs
        assert call_kwargs["repo_id"] == HF_REPO_ID
        assert "lancedb/*" in call_kwargs["allow_patterns"]
