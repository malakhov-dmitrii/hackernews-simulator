"""Tests for structural feature extraction.
Tests assert exact computed values for concrete inputs (not just "column exists").
"""
import pytest
import pandas as pd
import numpy as np


class TestTitleFeatures:
    def test_title_length_exact_values(self):
        from hackernews_simulator.features.structural import extract_title_features
        df = pd.DataFrame({"title": ["Short", "A much longer title with more words here"]})
        result = extract_title_features(df)
        assert result["title_length"].iloc[0] == 5
        assert result["title_length"].iloc[1] == 40

    def test_title_word_count_exact_values(self):
        from hackernews_simulator.features.structural import extract_title_features
        df = pd.DataFrame({"title": ["Two words", "One two three four"]})
        result = extract_title_features(df)
        assert result["title_word_count"].iloc[0] == 2
        assert result["title_word_count"].iloc[1] == 4

    def test_title_has_question_mark(self):
        from hackernews_simulator.features.structural import extract_title_features
        df = pd.DataFrame({"title": ["Is this a question?", "This is not"]})
        result = extract_title_features(df)
        assert result["title_has_question"].iloc[0] == 1
        assert result["title_has_question"].iloc[1] == 0

    def test_title_has_number(self):
        from hackernews_simulator.features.structural import extract_title_features
        df = pd.DataFrame({"title": ["Top 10 reasons", "No numbers here"]})
        result = extract_title_features(df)
        assert result["title_has_number"].iloc[0] == 1
        assert result["title_has_number"].iloc[1] == 0

    def test_title_starts_with_show_hn(self):
        from hackernews_simulator.features.structural import extract_title_features
        df = pd.DataFrame({"title": ["Show HN: My project", "Regular post"]})
        result = extract_title_features(df)
        assert result["is_show_hn"].iloc[0] == 1
        assert result["is_show_hn"].iloc[1] == 0

    def test_empty_title(self):
        from hackernews_simulator.features.structural import extract_title_features
        df = pd.DataFrame({"title": [""]})
        result = extract_title_features(df)
        assert result["title_length"].iloc[0] == 0
        assert result["title_word_count"].iloc[0] == 0
        assert result["title_has_question"].iloc[0] == 0
        assert result["is_show_hn"].iloc[0] == 0
        assert result["is_ask_hn"].iloc[0] == 0


class TestTemporalFeatures:
    def test_hour_of_day_exact(self):
        from hackernews_simulator.features.structural import extract_temporal_features
        df = pd.DataFrame({
            "time": pd.to_datetime(["2022-06-15 14:30:00", "2022-06-15 03:00:00"], utc=True)
        })
        result = extract_temporal_features(df)
        assert result["hour"].iloc[0] == 14
        assert result["hour"].iloc[1] == 3

    def test_day_of_week_exact(self):
        from hackernews_simulator.features.structural import extract_temporal_features
        # 2022-06-15 is a Wednesday (2)
        df = pd.DataFrame({
            "time": pd.to_datetime(["2022-06-15 14:30:00"], utc=True)
        })
        result = extract_temporal_features(df)
        assert result["day_of_week"].iloc[0] == 2

    def test_is_weekend(self):
        from hackernews_simulator.features.structural import extract_temporal_features
        df = pd.DataFrame({
            "time": pd.to_datetime(["2022-06-18 14:00:00", "2022-06-15 14:00:00"], utc=True)  # Sat, Wed
        })
        result = extract_temporal_features(df)
        assert result["is_weekend"].iloc[0] == 1
        assert result["is_weekend"].iloc[1] == 0


class TestUrlFeatures:
    def test_has_url(self):
        from hackernews_simulator.features.structural import extract_url_features
        df = pd.DataFrame({
            "url": ["https://example.com", ""],
            "domain": ["example.com", ""],
        })
        result = extract_url_features(df)
        assert result["has_url"].iloc[0] == 1
        assert result["has_url"].iloc[1] == 0

    def test_is_github(self):
        from hackernews_simulator.features.structural import extract_url_features
        df = pd.DataFrame({
            "url": ["https://github.com/repo", "https://blog.com"],
            "domain": ["github.com", "blog.com"],
        })
        result = extract_url_features(df)
        assert result["is_github"].iloc[0] == 1
        assert result["is_github"].iloc[1] == 0


class TestTextFeatures:
    def test_has_text_and_length_exact(self):
        from hackernews_simulator.features.structural import extract_text_presence_features
        df = pd.DataFrame({"clean_text": ["Hello world", ""]})
        result = extract_text_presence_features(df)
        assert result["has_text"].iloc[0] == 1
        assert result["has_text"].iloc[1] == 0
        assert result["text_length"].iloc[0] == 11
        assert result["text_length"].iloc[1] == 0


class TestCombineStructuralFeatures:
    def test_combine_returns_all_expected_columns(self, sample_stories_df):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.structural import extract_structural_features
        df = preprocess_stories(sample_stories_df)
        result = extract_structural_features(df)
        expected_cols = [
            "title_length", "title_word_count", "title_has_question",
            "title_has_number", "is_show_hn", "is_ask_hn",
            "hour", "day_of_week", "is_weekend",
            "has_url", "is_github", "has_text", "text_length",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_is_all_numeric(self, sample_stories_df):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.structural import extract_structural_features
        df = preprocess_stories(sample_stories_df)
        result = extract_structural_features(df)
        for col in result.columns:
            assert np.issubdtype(result[col].dtype, np.number), f"{col} is not numeric"

    def test_first_row_show_hn_values(self, sample_stories_df):
        """Verify exact feature values for the first story (Show HN post with github URL)."""
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.structural import extract_structural_features
        df = preprocess_stories(sample_stories_df)
        result = extract_structural_features(df)
        # First story: "Show HN: I built a tool to predict HN reactions", github.com URL
        assert result["is_show_hn"].iloc[0] == 1
        assert result["is_github"].iloc[0] == 1
        assert result["has_url"].iloc[0] == 1
        assert result["has_text"].iloc[0] == 1  # has text content
        assert result["title_word_count"].iloc[0] == 10
