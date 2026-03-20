"""Tests for combined feature pipeline.
Uses mock_embedding_model to avoid loading real model.
"""
import pytest
import numpy as np
import pandas as pd


class TestFeaturePipeline:
    def test_returns_numpy_array_and_names(self, sample_stories_df, mock_embedding_model):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.pipeline import build_feature_matrix
        df = preprocess_stories(sample_stories_df)
        X, feature_names = build_feature_matrix(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(feature_names, list)

    def test_correct_num_rows(self, sample_stories_df, mock_embedding_model):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.pipeline import build_feature_matrix
        df = preprocess_stories(sample_stories_df)
        X, _ = build_feature_matrix(df)
        assert X.shape[0] == len(df)

    def test_feature_count_structural_plus_embedding(self, sample_stories_df, mock_embedding_model):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.pipeline import build_feature_matrix
        df = preprocess_stories(sample_stories_df)
        X, feature_names = build_feature_matrix(df)
        # 15 structural + 384 embedding = 399
        assert X.shape[1] == 399
        assert len(feature_names) == 399

    def test_no_nan_in_output(self, sample_stories_df, mock_embedding_model):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.pipeline import build_feature_matrix
        df = preprocess_stories(sample_stories_df)
        X, _ = build_feature_matrix(df)
        assert not np.isnan(X).any()

    def test_feature_names_format(self, sample_stories_df, mock_embedding_model):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.pipeline import build_feature_matrix
        df = preprocess_stories(sample_stories_df)
        _, names = build_feature_matrix(df)
        assert all(isinstance(n, str) for n in names)
        # First 13 are structural names, last 384 are emb_0..emb_383
        assert names[0] == "title_length"
        assert names[-1] == "emb_383"

    def test_build_for_single_input(self, mock_embedding_model):
        """Test building features for a single prediction input (no score column)."""
        from hackernews_simulator.features.pipeline import build_feature_matrix_for_input
        title = "Show HN: My cool project"
        description = "I built this thing using Python and ML."
        X, feature_names = build_feature_matrix_for_input(title, description)
        assert X.shape == (1, 399)
        assert not np.isnan(X).any()
