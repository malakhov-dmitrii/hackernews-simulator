"""Tests for multi-variant creative comparison."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from dataclasses import dataclass


@pytest.fixture
def mock_models_and_simulator(tmp_path, mock_embedding_model):
    """Create a small trained HNSimulator for compare/suggest tests."""
    from hn_simulator.model.train import train_score_model, train_comment_count_model, save_model
    from hn_simulator.rag.index import build_story_index, build_comment_index
    from hn_simulator.simulator import HNSimulator
    import pandas as pd

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 397)).astype(np.float32)
    y_score = (X[:, 0] * 10 + rng.standard_normal(100)).clip(0, None)
    y_comments = (X[:, 1] * 5 + rng.standard_normal(100)).clip(0, None)
    names = [f"feat_{i}" for i in range(397)]

    score_model, _ = train_score_model(X[:80], y_score[:80], X[80:], y_score[80:], names)
    comment_model, _ = train_comment_count_model(X[:80], y_comments[:80], X[80:], y_comments[80:], names)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    save_model(score_model, models_dir / "score_model.txt")
    save_model(comment_model, models_dir / "comment_model.txt")

    stories_df = pd.DataFrame({
        "id": np.array([1], dtype=np.uint32),
        "title": ["Test story"],
        "url": ["https://example.com"],
        "clean_text": ["Some text"],
        "score": np.array([50], dtype=np.int32),
        "descendants": np.array([10], dtype=np.int32),
        "domain": ["example.com"],
        "post_type": ["regular"],
    })
    embeddings = rng.standard_normal((1, 384)).astype(np.float32)
    lancedb_dir = tmp_path / "lancedb"
    build_story_index(stories_df, embeddings, db_path=lancedb_dir)

    comments_df = pd.DataFrame({
        "id": np.array([101], dtype=np.uint32),
        "parent": np.array([1], dtype=np.uint32),
        "clean_text": ["This is a test comment"],
        "by": ["testuser"],
    })
    build_comment_index(comments_df, db_path=lancedb_dir)

    sim = HNSimulator(
        score_model_path=models_dir / "score_model.txt",
        comment_model_path=models_dir / "comment_model.txt",
        lancedb_path=lancedb_dir,
        claude_client=None,
    )
    return sim


class TestCompareVariants:
    def test_returns_ranked_results(self, mock_models_and_simulator):
        from hn_simulator.compare import compare_variants
        sim = mock_models_and_simulator

        variants = [
            {"title": "Show HN: I built an AI", "description": "Uses ML to predict..."},
            {"title": "Predicting HN: 47M posts analyzed", "description": "We analyzed the full dataset..."},
        ]
        results = compare_variants(sim, variants)
        assert len(results) == 2
        # Results sorted by predicted_score descending
        assert results[0].predicted_score >= results[1].predicted_score

    def test_preserves_variant_index(self, mock_models_and_simulator):
        from hn_simulator.compare import compare_variants
        sim = mock_models_and_simulator

        variants = [
            {"title": "Variant A", "description": "Desc A"},
            {"title": "Variant B", "description": "Desc B"},
        ]
        results = compare_variants(sim, variants)
        for r in results:
            assert hasattr(r, "variant_index")
            assert hasattr(r, "title")

    def test_single_variant(self, mock_models_and_simulator):
        from hn_simulator.compare import compare_variants
        sim = mock_models_and_simulator

        variants = [{"title": "Only one", "description": "Solo"}]
        results = compare_variants(sim, variants)
        assert len(results) == 1

    def test_empty_variants_raises(self, mock_models_and_simulator):
        from hn_simulator.compare import compare_variants
        sim = mock_models_and_simulator

        with pytest.raises(ValueError, match="at least one variant"):
            compare_variants(sim, [])


class TestComparisonExplanation:
    def test_generate_explanation_returns_string(self):
        from hn_simulator.compare import generate_comparison_explanation

        ranked = [
            {"variant_index": 0, "title": "Best variant", "predicted_score": 100.0, "reception_label": "viral"},
            {"variant_index": 1, "title": "Worst variant", "predicted_score": 10.0, "reception_label": "moderate"},
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Variant 1 wins because of technical framing.")]
        mock_client.messages.create.return_value = mock_response

        explanation = generate_comparison_explanation(ranked, client=mock_client)
        assert isinstance(explanation, str)
        assert len(explanation) > 10

    def test_works_without_client(self):
        from hn_simulator.compare import generate_comparison_explanation
        ranked = [
            {"variant_index": 0, "title": "Best", "predicted_score": 100.0, "reception_label": "viral"},
        ]
        explanation = generate_comparison_explanation(ranked, client=None)
        assert isinstance(explanation, str)  # Falls back to rule-based


class TestLoadVariantsFromYaml:
    def test_load_valid_yaml(self, tmp_path):
        from hn_simulator.compare import load_variants_from_file
        yaml_content = '''variants:
  - title: "Show HN: Test"
    description: "A test project"
  - title: "Ask HN: Test?"
    description: "Asking about test"
'''
        path = tmp_path / "variants.yaml"
        path.write_text(yaml_content)
        variants = load_variants_from_file(path)
        assert len(variants) == 2
        assert variants[0]["title"] == "Show HN: Test"
        assert variants[1]["description"] == "Asking about test"

    def test_load_invalid_yaml_raises(self, tmp_path):
        from hn_simulator.compare import load_variants_from_file
        path = tmp_path / "bad.yaml"
        path.write_text("not: valid: yaml: [")
        with pytest.raises((ValueError, Exception)):
            load_variants_from_file(path)

    def test_load_missing_title_raises(self, tmp_path):
        from hn_simulator.compare import load_variants_from_file
        yaml_content = '''variants:
  - description: "Missing title"
'''
        path = tmp_path / "notitle.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="title"):
            load_variants_from_file(path)
