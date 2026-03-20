"""HN comment generation via Claude CLI or API."""
import json
import logging
import re

from hn_simulator.config import CLAUDE_MODEL
from hn_simulator.comments.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


def parse_comments_response(text: str) -> list[dict]:
    """Parse JSON or markdown-wrapped JSON from Claude response.

    Validates each item has 'username' and 'comment' keys.
    Returns [] on any parse failure.
    """
    # Try direct JSON parse first
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return [item for item in data if "username" in item and "comment" in item]
        return []
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return [item for item in data if "username" in item and "comment" in item]
        except json.JSONDecodeError:
            pass

    return []


def generate_comments(
    title: str,
    description: str,
    predicted_score: float,
    predicted_label: str,
    similar_stories: list,
    similar_comments: list,
    client=None,
    num_comments: int = 5,
) -> list[dict]:
    """Generate HN-style comments via Claude CLI (headless) or mock client.

    If client is provided (e.g. in tests), uses the client's messages.create API.
    Otherwise, spawns headless Claude CLI — uses Claude Code subscription, no API key needed.
    Returns [] on any error.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        title=title,
        description=description,
        predicted_score=predicted_score,
        predicted_label=predicted_label,
        similar_stories=similar_stories,
        similar_comments=similar_comments,
        num_comments=num_comments,
    )

    try:
        if client is not None:
            # Mock client path (tests) or explicit API client
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text
        else:
            # Headless Claude CLI — no API key needed
            from hn_simulator.claude_runner import run_claude
            text = run_claude(
                prompt=user_prompt,
                system_prompt=system_prompt,
                timeout_seconds=120,
            )
        return parse_comments_response(text)
    except Exception as e:
        logger.warning("Comment generation error: %s", e)
        return []
