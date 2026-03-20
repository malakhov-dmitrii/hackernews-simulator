"""Data preprocessing — HTML stripping, domain extraction, post-type classification."""
from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup


def strip_html(text: str | None) -> str:
    """Strip HTML tags and decode HTML entities. Returns '' for None/empty."""
    if not text:
        return ""
    # Replace block-level tags with newlines before stripping so paragraphs
    # are separated rather than concatenated.
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(["p", "br"]):
        tag.insert_before("\n")
    result = soup.get_text()
    # Collapse leading/trailing whitespace but preserve interior newlines
    return result.strip()


def extract_domain(url: str | None) -> str:
    """Extract domain from URL, stripping www. prefix. Returns '' for bad input."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc
        if not netloc:
            return ""
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def classify_post_type(title: str | None) -> str:
    """Classify HN post type from title prefix."""
    if not title:
        return "regular"
    if title.startswith("Show HN:"):
        return "show_hn"
    if title.startswith("Ask HN:"):
        return "ask_hn"
    if title.startswith("Tell HN:"):
        return "tell_hn"
    if title.startswith("Launch HN:"):
        return "launch_hn"
    return "regular"


def preprocess_stories(df: pd.DataFrame, min_score: int = 0) -> pd.DataFrame:
    """Add clean_text, domain, post_type columns; optionally filter by min_score."""
    result = df.copy()
    result["clean_text"] = result["text"].apply(strip_html)
    result["domain"] = result["url"].apply(extract_domain)
    result["post_type"] = result["title"].apply(classify_post_type)
    if min_score > 0:
        result = result[result["score"] >= min_score]
    return result


def preprocess_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Add clean_text column; filter out rows where clean_text is empty."""
    result = df.copy()
    result["clean_text"] = result["text"].apply(strip_html)
    result = result[result["clean_text"] != ""]
    return result
