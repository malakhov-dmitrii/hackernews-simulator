"""Suggest mode — AI-generated variant optimization for HN post titles."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hackernews_simulator.simulator import HNSimulator

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
    previous_titles: list[str] | None = None,
) -> list[dict]:
    """Ask Claude to generate alternative title+description pairs for a post.

    Args:
        original: Dict with "title" and "description" keys.
        client: Anthropic client (required).
        num_suggestions: Number of variants to generate.
        previous_titles: Titles already tried — Claude will avoid re-suggesting them.

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

    if previous_titles:
        joined = ", ".join(f'"{t}"' for t in previous_titles)
        user_message += f"\n\nDo NOT suggest these titles (already tried): {joined}"

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
        from hackernews_simulator.claude_runner import run_claude
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


def iterative_optimize(
    simulator: "HNSimulator",
    original: dict,
    client=None,
    max_iterations: int = 5,
    min_improvement: float = 2.0,
    num_suggestions: int = 3,
) -> dict:
    """Iteratively suggest and score variants, converging on the best.

    Returns dict with:
      - best: {title, description, predicted_score}
      - all_variants: list of all scored variants
      - iterations: number of iterations run
      - improvement: total score improvement from original
    """
    orig_result = simulator.simulate(
        original.get("title", ""),
        original.get("description", ""),
        generate_comments=False,
    )
    original_score = orig_result.predicted_score

    current_best = {
        "title": original.get("title", ""),
        "description": original.get("description", ""),
        "predicted_score": original_score,
    }

    seen_titles: set[str] = {current_best["title"]}
    all_variants: list[dict] = []

    iterations_run = 0
    for _ in range(max_iterations):
        iterations_run += 1

        suggestions = suggest_variants(
            {"title": current_best["title"], "description": current_best["description"]},
            client=client,
            num_suggestions=num_suggestions,
            previous_titles=list(seen_titles),
        )

        # Filter out already-seen titles
        new_suggestions = [s for s in suggestions if s.get("title", "") not in seen_titles]

        if not new_suggestions:
            break

        # Score each new variant
        iteration_best = None
        iteration_best_score = current_best["predicted_score"]
        for s in new_suggestions:
            t = s.get("title", "")
            d = s.get("description", "")
            seen_titles.add(t)
            sim_result = simulator.simulate(t, d, generate_comments=False)
            variant = {
                "title": t,
                "description": d,
                "predicted_score": sim_result.predicted_score,
            }
            all_variants.append(variant)
            if sim_result.predicted_score > iteration_best_score:
                iteration_best_score = sim_result.predicted_score
                iteration_best = variant

        if iteration_best is not None and (iteration_best_score - current_best["predicted_score"]) >= min_improvement:
            current_best = iteration_best
        else:
            break

    improvement = current_best["predicted_score"] - original_score
    return {
        "best": current_best,
        "all_variants": all_variants,
        "iterations": iterations_run,
        "improvement": improvement,
    }
