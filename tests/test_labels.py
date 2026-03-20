"""Tests for reception label classification.
Includes boundary value tests with exact threshold assertions.
SCORE_THRESHOLDS: flop=3, low=15, moderate=100, hot=300, viral=300+
Classification: score <= 3 -> "flop", score <= 15 -> "low",
               score <= 100 -> "moderate", score <= 300 -> "hot", score > 300 -> "viral"
"""
import pytest


class TestReceptionLabel:
    def test_flop_well_below(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=1, comment_count=0) == "flop"

    def test_low_middle(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=10, comment_count=5) == "low"

    def test_moderate_middle(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=50, comment_count=30) == "moderate"

    def test_hot_middle(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=200, comment_count=80) == "hot"

    def test_viral_well_above(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=500, comment_count=200) == "viral"


class TestReceptionLabelBoundaryValues:
    """Exact boundary tests — thresholds: flop=3, low=15, moderate=100, hot=300, viral>300."""

    def test_score_0(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=0, comment_count=0) == "flop"

    def test_negative_score(self):
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=-5, comment_count=0) == "flop"

    def test_boundary_flop_inclusive(self):
        """score=3 is flop (score <= 3)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=3, comment_count=0) == "flop"

    def test_boundary_flop_to_low(self):
        """score=4 is low (score > 3 and <= 15)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=4, comment_count=2) == "low"

    def test_boundary_low_inclusive(self):
        """score=15 is low (score <= 15)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=15, comment_count=5) == "low"

    def test_boundary_low_to_moderate(self):
        """score=16 is moderate (score > 15 and <= 100)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=16, comment_count=8) == "moderate"

    def test_boundary_moderate_inclusive(self):
        """score=100 is moderate (score <= 100)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=100, comment_count=50) == "moderate"

    def test_boundary_moderate_to_hot(self):
        """score=101 is hot (score > 100 and <= 300)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=101, comment_count=60) == "hot"

    def test_boundary_hot_inclusive(self):
        """score=300 is hot (score <= 300)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=300, comment_count=150) == "hot"

    def test_boundary_hot_to_viral(self):
        """score=301 is viral (score > 300)."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=301, comment_count=160) == "viral"

    def test_float_score_at_boundary(self):
        """Float score 3.0 is flop, 3.5 is low."""
        from hackernews_simulator.model.labels import classify_reception
        assert classify_reception(score=3.0, comment_count=0) == "flop"
        assert classify_reception(score=3.5, comment_count=1) == "low"


class TestReceptionWithConfidence:
    def test_returns_tuple_of_three(self):
        from hackernews_simulator.model.labels import classify_reception_with_confidence
        label, confidence, distribution = classify_reception_with_confidence(
            predicted_score=50.0, predicted_comments=20.0
        )
        assert label in ("flop", "low", "moderate", "hot", "viral")
        assert 0.0 < confidence <= 1.0
        assert isinstance(distribution, dict)
        assert abs(sum(distribution.values()) - 1.0) < 0.01

    def test_high_score_viral_high_confidence(self):
        from hackernews_simulator.model.labels import classify_reception_with_confidence
        label, confidence, _ = classify_reception_with_confidence(
            predicted_score=1000.0, predicted_comments=500.0
        )
        assert label == "viral"
        assert confidence > 0.5


class TestReceptionDescription:
    def test_all_labels_have_descriptions(self):
        from hackernews_simulator.model.labels import get_reception_description
        for label in ("flop", "low", "moderate", "hot", "viral"):
            desc = get_reception_description(label)
            assert isinstance(desc, str)
            assert len(desc) > 10
