"""Headless Claude CLI runner — uses Claude Code subscription instead of API keys."""
import json
import os
import subprocess
from pathlib import Path


def run_claude(
    prompt: str,
    system_prompt: str = "",
    timeout_seconds: int = 120,
) -> str:
    """Run Claude CLI in headless mode and return the text response.

    Spawns `claude -p --output-format text` with bypassPermissions.
    Uses the user's Claude Code subscription (no API key needed).
    """
    args = [
        "claude",
        "-p",
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
    ]

    if system_prompt:
        args.extend(["--append-system-prompt", system_prompt])

    args.extend(["--", prompt])

    # Strip keys that interfere with child Claude process
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "CLAUDECODE")}

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=str(Path.home()),
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Claude CLI not found. Install it: https://docs.anthropic.com/en/docs/claude-code"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout_seconds}s")

    if proc.returncode != 0 and not proc.stdout.strip():
        stderr = proc.stderr[:500] if proc.stderr else "no stderr"
        raise RuntimeError(f"Claude CLI failed (exit {proc.returncode}): {stderr}")

    return proc.stdout.strip()
