"""HN comment generation via Claude API."""
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
    """Generate HN-style comments via Claude API.

    Creates anthropic.Anthropic() client if none provided.
    Returns [] on API exception.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(
                        title=title,
                        description=description,
                        predicted_score=predicted_score,
                        predicted_label=predicted_label,
                        similar_stories=similar_stories,
                        similar_comments=similar_comments,
                        num_comments=num_comments,
                    ),
                }
            ],
        )
        text = response.content[0].text
        return parse_comments_response(text)
    except Exception as e:
        logger.warning("Claude API error during comment generation: %s", e)
        return []
