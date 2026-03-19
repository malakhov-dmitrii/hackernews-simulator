"""Tests for reception label classification.
Includes boundary value tests with exact threshold assertions.
SCORE_THRESHOLDS: flop=3, moderate=15, hot=100, viral=300
Classification: score <= flop -> "flop", score <= moderate -> "moderate",
               score <= hot -> "hot", score > hot -> "viral"
"""
import pytest


class TestReceptionLabel:
    def test_flop_well_below(self):
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=1, comment_count=0) == "flop"

    def test_moderate_middle(self):
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=10, comment_count=5) == "moderate"

    def test_hot_middle(self):
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=50, comment_count=30) == "hot"

    def test_viral_well_above(self):
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=500, comment_count=200) == "viral"


class TestReceptionLabelBoundaryValues:
    """Exact boundary tests — thresholds: flop=3, moderate=15, hot=100, viral=300."""

    def test_score_0(self):
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=0, comment_count=0) == "flop"

    def test_negative_score(self):
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=-5, comment_count=0) == "flop"

    def test_boundary_flop_inclusive(self):
        """score=3 is flop (score <= 3)."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=3, comment_count=0) == "flop"

    def test_boundary_flop_to_moderate(self):
        """score=4 is moderate (score > 3 and <= 15)."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=4, comment_count=2) == "moderate"

    def test_boundary_moderate_inclusive(self):
        """score=15 is moderate (score <= 15)."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=15, comment_count=5) == "moderate"

    def test_boundary_moderate_to_hot(self):
        """score=16 is hot (score > 15 and <= 100)."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=16, comment_count=8) == "hot"

    def test_boundary_hot_inclusive(self):
        """score=100 is hot (score <= 100)."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=100, comment_count=50) == "hot"

    def test_boundary_hot_to_viral(self):
        """score=101 is viral (score > 100)."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=101, comment_count=60) == "viral"

    def test_boundary_viral_threshold(self):
        """score=300 is viral."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=300, comment_count=150) == "viral"

    def test_float_score_at_boundary(self):
        """Float score 3.0 is flop, 3.5 is moderate."""
        from hn_simulator.model.labels import classify_reception
        assert classify_reception(score=3.0, comment_count=0) == "flop"
        assert classify_reception(score=3.5, comment_count=1) == "moderate"


class TestReceptionWithConfidence:
    def test_returns_tuple_of_three(self):
        from hn_simulator.model.labels import classify_reception_with_confidence
        label, confidence, distribution = classify_reception_with_confidence(
            predicted_score=50.0, predicted_comments=20.0
        )
        assert label in ("flop", "moderate", "hot", "viral")
        assert 0.0 < confidence <= 1.0
        assert isinstance(distribution, dict)
        assert abs(sum(distribution.values()) - 1.0) < 0.01

    def test_high_score_viral_high_confidence(self):
        from hn_simulator.model.labels import classify_reception_with_confidence
        label, confidence, _ = classify_reception_with_confidence(
            predicted_score=1000.0, predicted_comments=500.0
        )
        assert label == "viral"
        assert confidence > 0.5


class TestReceptionDescription:
    def test_all_labels_have_descriptions(self):
        from hn_simulator.model.labels import get_reception_description
        for label in ("flop", "moderate", "hot", "viral"):
            desc = get_reception_description(label)
            assert isinstance(desc, str)
            assert len(desc) > 10
