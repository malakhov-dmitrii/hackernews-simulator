"""Tests for rich_output module."""
from __future__ import annotations

from types import SimpleNamespace

from hackernews_simulator.rich_output import label_color, print_comparison, print_prediction


def test_label_color_flop():
    assert label_color("flop") == "red"


def test_label_color_viral():
    assert label_color("viral") == "magenta"


def test_label_color_unknown():
    assert label_color("unknown") == "white"


def _make_result(**kwargs):
    defaults = dict(
        predicted_score=42.0,
        predicted_comments=5.0,
        reception_label="moderate",
        confidence=0.75,
        percentile=None,
        expected_score=None,
        label_distribution=None,
        shap_features=None,
        time_recommendation=None,
        simulated_comments=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_print_prediction_does_not_raise():
    result = _make_result()
    print_prediction("Show HN: Test project", result)


def test_print_prediction_full_does_not_raise():
    result = _make_result(
        percentile=10.5,
        expected_score=120.0,
        label_distribution={"flop": 0.1, "low": 0.2, "moderate": 0.4, "hot": 0.2, "viral": 0.1},
        shap_features=[{"feature": "title_length", "direction": "up", "importance": 0.5}],
        time_recommendation="Post on Tuesday morning for best results.",
        simulated_comments=[{"username": "testuser", "tone": "skeptical", "comment": "Interesting."}],
    )
    print_prediction("Show HN: Full test", result)


def test_print_comparison_does_not_raise():
    results = [
        {"title": "Option A", "predicted_score": 100, "reception_label": "hot"},
        {"title": "Option B", "predicted_score": 20, "reception_label": "flop"},
    ]
    print_comparison(results)
