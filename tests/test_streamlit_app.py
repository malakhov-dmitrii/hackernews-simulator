"""Tests for streamlit_app helper functions (no Streamlit runtime needed)."""
from __future__ import annotations


def test_label_colors_has_all_five_labels():
    from streamlit_app import LABEL_COLORS
    assert set(LABEL_COLORS.keys()) == {"flop", "low", "moderate", "hot", "viral"}


def test_format_score_display_returns_string_with_points():
    from streamlit_app import format_score_display
    result = format_score_display(42.7)
    assert isinstance(result, str)
    assert "points" in result


def test_format_score_display_rounds_down():
    from streamlit_app import format_score_display
    assert "42" in format_score_display(42.9)


def test_load_simulator_function_exists():
    import streamlit_app
    assert callable(streamlit_app.load_simulator)
