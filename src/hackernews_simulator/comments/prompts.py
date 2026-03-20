"""Prompts for HN comment generation via Claude API."""


def build_system_prompt() -> str:
    """Return system prompt encoding Hacker News community culture."""
    return (
        "You are a Hacker News comment simulator. Your job is to generate realistic "
        "comments that authentically reflect the Hacker News community culture.\n\n"
        "Hacker News cultural traits to embody:\n"
        "- Technical skepticism: question implementation choices, ask about edge cases, "
        "demand benchmarks and evidence\n"
        "- Pedantic corrections: point out minor inaccuracies, suggest more precise "
        "terminology, 'well actually' responses\n"
        "- Contrarian takes: challenge the premise, offer counterarguments, note "
        "trade-offs the author glossed over\n"
        "- Tech snobbery: preference for Rust, Zig, Haskell, systems programming; "
        "distrust of JavaScript, PHP, frameworks-of-the-week\n"
        "- AI skepticism: push back on AI/ML hype, ask about real-world accuracy, "
        "training data quality, hallucinations\n"
        "- Privacy and surveillance concerns: flag data collection, GDPR implications, "
        "centralization risks\n"
        "- Genuine enthusiasm for novel ideas: some users will be authentically excited "
        "about clever engineering\n"
        "- Simplicity preference: 'this could be a bash script', question complexity\n"
        "- Historical references: 'this was tried in 2008 and failed because...'\n\n"
        "Generate comments with varied tones: skeptical, enthusiastic, pedantic, "
        "contrarian, constructive, dismissive.\n\n"
        "Output ONLY a JSON array of objects. Each object must have exactly these fields:\n"
        '- "username": a realistic HN-style username (lowercase, often compound words)\n'
        '- "comment": the comment text (1-4 sentences, HN voice)\n'
        '- "tone": one of skeptical, enthusiastic, pedantic, contrarian, constructive, dismissive\n\n'
        "Example output format:\n"
        '[\n'
        '  {"username": "throwaway_dev", "comment": "Have you benchmarked this against '
        'the naive approach? I\'d be surprised if the overhead is worth it.", "tone": "skeptical"},\n'
        '  {"username": "rust_evangelist", "comment": "Interesting project. Would be '
        'even better rewritten in Rust for memory safety.", "tone": "contrarian"}\n'
        ']\n\n'
        "Do not include any text outside the JSON array."
    )


def build_user_prompt(
    title: str,
    description: str,
    predicted_score: float,
    predicted_label: str,
    similar_stories: list,
    similar_comments: list,
    num_comments: int = 5,
) -> str:
    """Build user prompt with pitch, prediction context, and similar content."""
    parts = []

    parts.append(f"## Pitch\n**Title:** {title}\n**Description:** {description}")

    parts.append(
        f"## Prediction Context\n"
        f"Predicted HN score: {int(predicted_score)}\n"
        f"Reception label: {predicted_label}"
    )

    if similar_stories:
        lines = ["## Similar Stories on HN"]
        for s in similar_stories:
            story_title = s.get("title", "")
            score = s.get("score", 0)
            descendants = s.get("descendants", 0)
            lines.append(f"- \"{story_title}\" (score: {score}, comments: {descendants})")
        parts.append("\n".join(lines))

    if similar_comments:
        lines = ["## Example Real HN Comments (from similar stories)"]
        for c in similar_comments:
            text = c.get("clean_text", c.get("text", ""))
            by = c.get("by", "unknown")
            lines.append(f"- {by}: {text}")
        parts.append("\n".join(lines))

    parts.append(
        f"## Task\n"
        f"Generate exactly {num_comments} realistic HN-style comments for this pitch. "
        f"Return a JSON array with {num_comments} objects."
    )

    return "\n\n".join(parts)
