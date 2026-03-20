"""Tests for comment generation.
Uses mocked Claude API to avoid real API calls in tests.
Includes error path tests for API failures.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestGenerateComments:
    def test_returns_list_of_comments(self):
        from hackernews_simulator.comments.generate import generate_comments
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""[
            {"username": "skeptic_dev", "comment": "Have you considered the scaling implications?", "tone": "skeptical"},
            {"username": "ml_enthusiast", "comment": "This is really neat! What's the training set size?", "tone": "enthusiastic"}
        ]""")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        comments = generate_comments(
            title="Show HN: My ML Project",
            description="A tool for predicting HN reactions",
            predicted_score=50.0,
            predicted_label="hot",
            similar_stories=[],
            similar_comments=[],
            client=mock_client,
        )
        assert isinstance(comments, list)
        assert len(comments) == 2
        assert "username" in comments[0]
        assert "comment" in comments[0]

    def test_handles_malformed_api_response(self):
        from hackernews_simulator.comments.generate import generate_comments
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not valid JSON")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        comments = generate_comments(
            title="Test",
            description="Desc",
            predicted_score=10.0,
            predicted_label="moderate",
            similar_stories=[],
            similar_comments=[],
            client=mock_client,
        )
        assert isinstance(comments, list)
        assert comments == []

    def test_calls_claude_with_correct_model(self):
        from hackernews_simulator.comments.generate import generate_comments
        from hackernews_simulator.config import CLAUDE_MODEL
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[]")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        generate_comments(
            title="Test", description="Desc",
            predicted_score=10.0, predicted_label="moderate",
            similar_stories=[], similar_comments=[],
            client=mock_client,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == CLAUDE_MODEL

    def test_handles_api_exception(self):
        """Error path: Claude API raises an exception."""
        from hackernews_simulator.comments.generate import generate_comments
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limited")

        comments = generate_comments(
            title="Test", description="Desc",
            predicted_score=10.0, predicted_label="moderate",
            similar_stories=[], similar_comments=[],
            client=mock_client,
        )
        # Should return empty list, not crash
        assert isinstance(comments, list)
        assert comments == []


class TestParseComments:
    def test_parse_valid_json(self):
        from hackernews_simulator.comments.generate import parse_comments_response
        text = '[{"username": "user1", "comment": "Great!", "tone": "positive"}]'
        result = parse_comments_response(text)
        assert len(result) == 1

    def test_parse_json_in_markdown_block(self):
        from hackernews_simulator.comments.generate import parse_comments_response
        text = '```json\n[{"username": "user1", "comment": "Great!", "tone": "positive"}]\n```'
        result = parse_comments_response(text)
        assert len(result) == 1

    def test_parse_invalid_returns_empty(self):
        from hackernews_simulator.comments.generate import parse_comments_response
        result = parse_comments_response("not json at all")
        assert result == []

    def test_parse_missing_keys_filters_invalid(self):
        from hackernews_simulator.comments.generate import parse_comments_response
        text = '[{"username": "user1"}, {"username": "user2", "comment": "Valid"}]'
        result = parse_comments_response(text)
        # Only the item with both username and comment should remain
        assert len(result) == 1
        assert result[0]["username"] == "user2"
