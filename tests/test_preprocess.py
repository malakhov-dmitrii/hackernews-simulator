"""Tests for data preprocessing.
All fixtures use open-index schema: id, type, by, time, text, score, title,
url, descendants, parent, dead, deleted.
"""
import pytest
import pandas as pd
import numpy as np


class TestStripHtml:
    def test_strips_p_tags(self):
        from hn_simulator.data.preprocess import strip_html
        assert strip_html("<p>Hello world</p>") == "Hello world"

    def test_strips_nested_tags(self):
        from hn_simulator.data.preprocess import strip_html
        assert strip_html("<p>Some <b>bold</b> and <i>italic</i></p>") == "Some bold and italic"

    def test_decodes_html_entities(self):
        from hn_simulator.data.preprocess import strip_html
        assert strip_html("I&#x27;m happy &amp; excited") == "I'm happy & excited"

    def test_handles_none(self):
        from hn_simulator.data.preprocess import strip_html
        assert strip_html(None) == ""

    def test_handles_empty_string(self):
        from hn_simulator.data.preprocess import strip_html
        assert strip_html("") == ""

    def test_preserves_plain_text(self):
        from hn_simulator.data.preprocess import strip_html
        assert strip_html("Just plain text") == "Just plain text"

    def test_strips_links_keeps_text(self):
        from hn_simulator.data.preprocess import strip_html
        result = strip_html('<a href="https://example.com">click here</a>')
        assert result == "click here"

    def test_replaces_br_and_p_with_newlines(self):
        from hn_simulator.data.preprocess import strip_html
        result = strip_html("<p>First paragraph</p><p>Second paragraph</p>")
        assert "First paragraph" in result
        assert "Second paragraph" in result


class TestExtractDomain:
    def test_extracts_github(self):
        from hn_simulator.data.preprocess import extract_domain
        assert extract_domain("https://github.com/user/repo") == "github.com"

    def test_extracts_with_www(self):
        from hn_simulator.data.preprocess import extract_domain
        assert extract_domain("https://www.nytimes.com/article") == "nytimes.com"

    def test_handles_empty(self):
        from hn_simulator.data.preprocess import extract_domain
        assert extract_domain("") == ""

    def test_handles_none(self):
        from hn_simulator.data.preprocess import extract_domain
        assert extract_domain(None) == ""

    def test_handles_malformed_url(self):
        from hn_simulator.data.preprocess import extract_domain
        assert extract_domain("not a url") == ""


class TestClassifyPostType:
    def test_show_hn(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type("Show HN: My cool project") == "show_hn"

    def test_ask_hn(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type("Ask HN: What are you working on?") == "ask_hn"

    def test_tell_hn(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type("Tell HN: I'm leaving my job") == "tell_hn"

    def test_launch_hn(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type("Launch HN: My startup is live") == "launch_hn"

    def test_regular(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type("Why Rust is great") == "regular"

    def test_empty_title(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type("") == "regular"

    def test_none_title(self):
        from hn_simulator.data.preprocess import classify_post_type
        assert classify_post_type(None) == "regular"


class TestPreprocessStories:
    """Uses open-index schema columns: id, type, title, url, text, score,
    descendants, time, by, dead, deleted."""

    def test_adds_clean_text_column(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        result = preprocess_stories(sample_stories_df)
        assert "clean_text" in result.columns
        # HTML should be stripped
        assert "<p>" not in str(result["clean_text"].iloc[0])

    def test_adds_domain_column(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        result = preprocess_stories(sample_stories_df)
        assert "domain" in result.columns
        assert result["domain"].iloc[0] == "github.com"

    def test_adds_post_type_column(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        result = preprocess_stories(sample_stories_df)
        assert "post_type" in result.columns
        assert result["post_type"].iloc[0] == "show_hn"
        assert result["post_type"].iloc[2] == "ask_hn"

    def test_filters_by_min_score(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        result = preprocess_stories(sample_stories_df, min_score=3)
        assert all(result["score"] >= 3)

    def test_preserves_row_count_without_filter(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        result = preprocess_stories(sample_stories_df, min_score=0)
        assert len(result) == len(sample_stories_df)

    def test_preserves_original_columns(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        result = preprocess_stories(sample_stories_df)
        # Must keep open-index columns
        for col in ["id", "score", "descendants", "time", "by"]:
            assert col in result.columns


class TestPreprocessComments:
    """Uses open-index schema: id, type, parent, text, by, time."""

    def test_strips_html_from_comments(self, sample_comments_df):
        from hn_simulator.data.preprocess import preprocess_comments
        result = preprocess_comments(sample_comments_df)
        assert "clean_text" in result.columns
        assert "<p>" not in str(result["clean_text"].iloc[0])

    def test_filters_empty_comments(self):
        from hn_simulator.data.preprocess import preprocess_comments
        df = pd.DataFrame({
            "id": np.array([101, 102], dtype=np.uint32),
            "type": np.array([2, 2], dtype=np.int8),
            "parent": np.array([1, 2], dtype=np.uint32),
            "text": ["<p>Real comment</p>", ""],
            "by": ["user1", "user2"],
            "time": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "dead": np.array([0, 0], dtype=np.uint8),
            "deleted": np.array([0, 0], dtype=np.uint8),
        })
        result = preprocess_comments(df)
        assert len(result) == 1

    def test_preserves_parent_as_uint32(self, sample_comments_df):
        from hn_simulator.data.preprocess import preprocess_comments
        result = preprocess_comments(sample_comments_df)
        assert "parent" in result.columns
