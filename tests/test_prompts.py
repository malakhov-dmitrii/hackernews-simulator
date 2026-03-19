"""Tests for HN comment generation prompts."""
import pytest


class TestSystemPrompt:
    def test_system_prompt_mentions_hn(self):
        from hn_simulator.comments.prompts import build_system_prompt
        prompt = build_system_prompt()
        assert "Hacker News" in prompt

    def test_system_prompt_mentions_culture(self):
        from hn_simulator.comments.prompts import build_system_prompt
        prompt = build_system_prompt()
        assert any(word in prompt.lower() for word in ["skeptic", "technical", "contrarian", "pedantic"])

    def test_system_prompt_has_json_instruction(self):
        from hn_simulator.comments.prompts import build_system_prompt
        prompt = build_system_prompt()
        assert "json" in prompt.lower() or "JSON" in prompt


class TestUserPrompt:
    def test_includes_title(self):
        from hn_simulator.comments.prompts import build_user_prompt
        prompt = build_user_prompt(
            title="Show HN: My Project",
            description="A cool project",
            predicted_score=50,
            predicted_label="hot",
            similar_stories=[],
            similar_comments=[],
        )
        assert "Show HN: My Project" in prompt

    def test_includes_prediction_context(self):
        from hn_simulator.comments.prompts import build_user_prompt
        prompt = build_user_prompt(
            title="Test",
            description="Desc",
            predicted_score=150,
            predicted_label="viral",
            similar_stories=[],
            similar_comments=[],
        )
        assert "150" in prompt
        assert "viral" in prompt

    def test_includes_similar_stories(self):
        from hn_simulator.comments.prompts import build_user_prompt
        similar = [
            {"title": "Similar Project", "score": 200, "descendants": 50},
        ]
        prompt = build_user_prompt(
            title="Test",
            description="Desc",
            predicted_score=50,
            predicted_label="hot",
            similar_stories=similar,
            similar_comments=[],
        )
        assert "Similar Project" in prompt

    def test_includes_example_comments(self):
        from hn_simulator.comments.prompts import build_user_prompt
        comments = [
            {"clean_text": "This is really cool!", "by": "user1"},
        ]
        prompt = build_user_prompt(
            title="Test",
            description="Desc",
            predicted_score=50,
            predicted_label="hot",
            similar_stories=[],
            similar_comments=comments,
        )
        assert "This is really cool!" in prompt

    def test_requests_correct_comment_count(self):
        from hn_simulator.comments.prompts import build_user_prompt
        prompt = build_user_prompt(
            title="Test",
            description="Desc",
            predicted_score=50,
            predicted_label="hot",
            similar_stories=[],
            similar_comments=[],
            num_comments=5,
        )
        assert "5" in prompt
