"""Compare mode — multi-variant creative testing for HN post optimization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from hn_simulator.simulator import HNSimulator

from hn_simulator.simulator import SimulationResult


@dataclass
class ComparisonResult(SimulationResult):
    """SimulationResult extended with variant metadata."""

    variant_index: int = 0
    title: str = ""


def compare_variants(
    simulator: "HNSimulator",
    variants: list[dict],
    generate_comments: bool = True,
) -> list[ComparisonResult]:
    """Run simulation for each variant and return results ranked by predicted_score.

    Args:
        simulator: Initialized HNSimulator instance.
        variants: List of dicts with "title" and optional "description".
        generate_comments: Passed through to simulator.simulate().

    Returns:
        List of ComparisonResult sorted by predicted_score descending.

    Raises:
        ValueError: If variants is empty.
    """
    if not variants:
        raise ValueError("compare_variants requires at least one variant")

    results: list[ComparisonResult] = []
    for idx, variant in enumerate(variants):
        title = variant["title"]
        description = variant.get("description", "")
        sim_result = simulator.simulate(title, description, generate_comments=generate_comments)
        result = ComparisonResult(
            predicted_score=sim_result.predicted_score,
            predicted_comments=sim_result.predicted_comments,
            reception_label=sim_result.reception_label,
            confidence=sim_result.confidence,
            label_distribution=sim_result.label_distribution,
            simulated_comments=sim_result.simulated_comments,
            similar_stories=sim_result.similar_stories,
            variant_index=idx,
            title=title,
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_score, reverse=True)
    return results


def generate_comparison_explanation(ranked: list[dict], client=None) -> str:
    """Explain why the top-ranked variant scored higher than others.

    Args:
        ranked: List of dicts (variant_index, title, predicted_score, reception_label),
                sorted best-first.
        client: Optional Anthropic client. If None, returns rule-based summary.

    Returns:
        Explanation string.
    """
    if not ranked:
        return "No variants to compare."

    best = ranked[0]

    if client is None:
        # Rule-based fallback
        return (
            f"'{best['title']}' is predicted to perform best "
            f"(score: {best['predicted_score']:.1f}, label: {best['reception_label']})."
        )

    # Build a compact summary for Claude
    summary_lines = []
    for i, r in enumerate(ranked):
        summary_lines.append(
            f"{i + 1}. \"{r['title']}\" — score: {r['predicted_score']:.1f}, label: {r['reception_label']}"
        )
    summary = "\n".join(summary_lines)

    prompt = (
        "You are an expert on Hacker News culture and post optimization.\n\n"
        "Here are post variants ranked by predicted HN score:\n"
        f"{summary}\n\n"
        "Explain in 2-3 sentences why variant #1 scored higher and suggest one concrete "
        "improvement for each lower-ranked variant."
    )

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:
        # Fallback to Claude CLI if client call fails
        from hn_simulator.claude_runner import run_claude
        return run_claude(prompt=prompt, timeout_seconds=60)


def load_variants_from_file(path: Path | str) -> list[dict]:
    """Load variants from a YAML file.

    Expected format:
        variants:
          - title: "Show HN: ..."
            description: "..."

    Args:
        path: Path to YAML file.

    Returns:
        List of variant dicts.

    Raises:
        ValueError: If YAML is malformed, missing "variants" key, or any variant
                    is missing "title".
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict) or "variants" not in data:
        raise ValueError(f"YAML file must have a top-level 'variants' key, got: {path}")

    variants = data["variants"]
    if not isinstance(variants, list):
        raise ValueError(f"'variants' must be a list, got {type(variants)}")

    for i, v in enumerate(variants):
        if not isinstance(v, dict) or "title" not in v:
            raise ValueError(
                f"Each variant must have a 'title' field. Variant #{i} is missing 'title'."
            )

    return variants
