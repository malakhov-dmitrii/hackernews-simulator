"""Suggest mode — AI-generated variant optimization for HN post titles."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hn_simulator.simulator import HNSimulator

_SYSTEM_PROMPT = (
    "You are an expert at crafting Hacker News titles. "
    "Given a post pitch, generate alternative title+description pairs optimized for HN engagement. "
    "Consider: technical framing, Show HN vs Ask HN, avoiding AI hype triggers, concrete numbers. "
    "Return ONLY a JSON array of objects with 'title' and 'description' keys. No extra text."
)

_HN_CULTURE_CONTEXT = (
    "HN culture tips: prefer technical substance over marketing, "
    "use 'Show HN:' for projects you built, 'Ask HN:' for questions, "
    "lead with concrete results/numbers when possible, avoid superlatives."
)


def suggest_variants(
    original: dict,
    client,
    num_suggestions: int = 5,
) -> list[dict]:
    """Ask Claude to generate alternative title+description pairs for a post.

    Args:
        original: Dict with "title" and "description" keys.
        client: Anthropic client (required).
        num_suggestions: Number of variants to generate.

    Returns:
        List of dicts with "title" and "description". Empty list on malformed response.

    Raises:
        ValueError: If client is None.
    """
    title = original.get("title", "")
    description = original.get("description", "")

    user_message = (
        f"Original title: {title}\n"
        f"Original description: {description}\n\n"
        f"{_HN_CULTURE_CONTEXT}\n\n"
        f"Generate {num_suggestions} alternative title+description pairs as a JSON array."
    )

    if client is not None:
        # Mock client path (tests) or explicit API client
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text
    else:
        # Headless Claude CLI — no API key needed
        from hn_simulator.claude_runner import run_claude
        raw_text = run_claude(
            prompt=user_message,
            system_prompt=_SYSTEM_PROMPT,
            timeout_seconds=90,
        )
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            return []
        # Filter to only items that have at least a title key
        return [item for item in parsed if isinstance(item, dict) and "title" in item]
    except (json.JSONDecodeError, Exception):
        return []


def suggest_and_score(
    simulator: "HNSimulator",
    original: dict,
    client,
    num_suggestions: int = 5,
    generate_comments: bool = False,
) -> list[dict]:
    """Generate variant suggestions and score each alongside the original.

    Args:
        simulator: Initialized HNSimulator instance.
        original: Dict with "title" and "description" keys.
        client: Anthropic client (passed to suggest_variants).
        num_suggestions: How many AI suggestions to generate.
        generate_comments: Whether to generate simulated comments per variant.

    Returns:
        List of dicts sorted by predicted_score descending. Each dict has:
        title, description, predicted_score, predicted_comments, reception_label, is_original.
    """
    suggestions = suggest_variants(original, client=client, num_suggestions=num_suggestions)

    all_variants = [
        {"title": original.get("title", ""), "description": original.get("description", ""), "is_original": True},
    ] + [
        {"title": s.get("title", ""), "description": s.get("description", ""), "is_original": False}
        for s in suggestions
    ]

    results = []
    for variant in all_variants:
        sim_result = simulator.simulate(
            variant["title"],
            variant["description"],
            generate_comments=generate_comments,
        )
        results.append({
            "title": variant["title"],
            "description": variant["description"],
            "predicted_score": sim_result.predicted_score,
            "predicted_comments": sim_result.predicted_comments,
            "reception_label": sim_result.reception_label,
            "is_original": variant["is_original"],
        })

    results.sort(key=lambda r: r["predicted_score"], reverse=True)
    return results
