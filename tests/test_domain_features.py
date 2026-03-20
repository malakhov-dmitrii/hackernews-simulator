"""Tests for domain reputation features."""
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def domain_stats():
    """Sample domain stats dict."""
    return {
        "github.com": {"avg_score": 120.5, "post_count": 5000},
        "blog.example.com": {"avg_score": 25.0, "post_count": 50},
    }


@pytest.fixture
def domain_stats_file(tmp_path, domain_stats):
    """Write domain stats to a temp JSON file."""
    path = tmp_path / "domain_stats.json"
    path.write_text(json.dumps(domain_stats))
    return path


class TestComputeDomainStats:
    def test_computes_bayesian_smoothed_avg(self):
        from hackernews_simulator.features.structural import compute_domain_stats
        df = pd.DataFrame({
            "domain": ["github.com", "github.com", "github.com", "blog.com"],
            "score": np.array([100, 200, 300, 10], dtype=np.int32),
        })
        stats = compute_domain_stats(df, k=10, global_mean=50.0)
        # github.com: n=3, mean=200, smoothed = (3*200 + 10*50) / (3+10) = 1100/13 ≈ 84.6
        assert abs(stats["github.com"]["avg_score"] - 84.615) < 0.1
        assert stats["github.com"]["post_count"] == 3
        # blog.com: n=1, mean=10, smoothed = (1*10 + 10*50) / (1+10) = 510/11 ≈ 46.36
        assert abs(stats["blog.com"]["avg_score"] - 46.36) < 0.1
        assert stats["blog.com"]["post_count"] == 1

    def test_empty_dataframe_returns_empty_dict(self):
        from hackernews_simulator.features.structural import compute_domain_stats
        df = pd.DataFrame({"domain": pd.Series(dtype=str), "score": pd.Series(dtype=np.int32)})
        stats = compute_domain_stats(df, k=10, global_mean=50.0)
        assert stats == {}


class TestExtractDomainReputationFeatures:
    def test_known_domain_gets_stats(self, domain_stats):
        from hackernews_simulator.features.structural import extract_domain_reputation_features
        df = pd.DataFrame({"domain": ["github.com"]})
        result = extract_domain_reputation_features(df, domain_stats, global_mean=47.57)
        assert result["domain_avg_score"].iloc[0] == pytest.approx(120.5)
        assert result["domain_post_count"].iloc[0] == 5000

    def test_unknown_domain_gets_global_mean(self, domain_stats):
        from hackernews_simulator.features.structural import extract_domain_reputation_features
        df = pd.DataFrame({"domain": ["unknown.org"]})
        result = extract_domain_reputation_features(df, domain_stats, global_mean=47.57)
        assert result["domain_avg_score"].iloc[0] == pytest.approx(47.57)
        assert result["domain_post_count"].iloc[0] == 0

    def test_empty_domain_gets_global_mean(self, domain_stats):
        from hackernews_simulator.features.structural import extract_domain_reputation_features
        df = pd.DataFrame({"domain": [""]})
        result = extract_domain_reputation_features(df, domain_stats, global_mean=47.57)
        assert result["domain_avg_score"].iloc[0] == pytest.approx(47.57)
        assert result["domain_post_count"].iloc[0] == 0


class TestStructuralFeaturesIncludesDomain:
    def test_extract_structural_features_has_15_columns(self, sample_stories_df):
        from hackernews_simulator.data.preprocess import preprocess_stories
        from hackernews_simulator.features.structural import extract_structural_features
        df = preprocess_stories(sample_stories_df)
        result = extract_structural_features(df)
        assert "domain_avg_score" in result.columns
        assert "domain_post_count" in result.columns
        assert len(result.columns) == 15
