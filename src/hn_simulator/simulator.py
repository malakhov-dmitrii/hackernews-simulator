"""HN Simulator orchestrator — coordinates models, RAG, and comment generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hn_simulator.features.pipeline import build_feature_matrix_for_input
from hn_simulator.features.text import embed_texts
from hn_simulator.model.labels import classify_reception_with_confidence, expected_score_from_probs
from hn_simulator.model.predict import predict_score
from hn_simulator.model.train import load_model
from hn_simulator.rag.retrieve import retrieve_comments_for_story, retrieve_similar_stories


@dataclass
class SimulationResult:
    """Result of a full HN simulation run."""

    predicted_score: float
    predicted_comments: float
    reception_label: str
    confidence: float
    label_distribution: dict[str, float]
    simulated_comments: list[dict] = field(default_factory=list)
    similar_stories: list[dict] = field(default_factory=list)
    percentile: float | None = None
    shap_features: list[dict] = field(default_factory=list)
    time_recommendation: str = ""
    expected_score: float | None = None

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization."""
        return {
            "predicted_score": self.predicted_score,
            "predicted_comments": self.predicted_comments,
            "reception_label": self.reception_label,
            "confidence": self.confidence,
            "label_distribution": self.label_distribution,
            "simulated_comments": self.simulated_comments,
            "similar_stories": self.similar_stories,
            "percentile": self.percentile,
            "shap_features": self.shap_features,
            "time_recommendation": self.time_recommendation,
            "expected_score": self.expected_score,
        }


class HNSimulator:
    """Orchestrates HN story simulation: feature extraction, prediction, RAG, comment generation."""

    def __init__(
        self,
        score_model_path: Path | str,
        comment_model_path: Path | str,
        lancedb_path: Path | str,
        claude_client=None,
        multiclass_model_path: Path | str | None = None,
        sorted_scores_path: Path | str | None = None,
        time_stats_path: Path | str | None = None,
    ) -> None:
        self.score_model = load_model(Path(score_model_path))
        self.comment_model = load_model(Path(comment_model_path))
        self.lancedb_path = Path(lancedb_path)
        self.claude_client = claude_client

        # Optional v2 models/data
        self.multiclass_model = None
        if multiclass_model_path is not None:
            try:
                self.multiclass_model = load_model(Path(multiclass_model_path))
            except FileNotFoundError:
                pass

        self.sorted_scores: np.ndarray | None = None
        if sorted_scores_path is not None:
            try:
                from hn_simulator.model.calibrate import load_sorted_scores
                self.sorted_scores = load_sorted_scores(Path(sorted_scores_path))
            except (FileNotFoundError, Exception):
                pass

        self.time_stats: tuple | None = None
        if time_stats_path is not None:
            try:
                from hn_simulator.model.calibrate import load_time_stats
                self.time_stats = load_time_stats(Path(time_stats_path))
            except (FileNotFoundError, Exception):
                pass

    def simulate(
        self,
        title: str,
        description: str = "",
        generate_comments: bool = True,
    ) -> SimulationResult:
        """Run full simulation for a story pitch.

        Args:
            title: Story title (e.g. "Show HN: My project").
            description: Optional body text / description.
            generate_comments: If True and claude_client is set, generate AI comments.

        Returns:
            SimulationResult with all fields populated.
        """
        # 1. Build feature matrix
        X, _ = build_feature_matrix_for_input(title, description)

        # 2. Predict score and comment count
        predicted_score = predict_score(self.score_model, X)
        predicted_comments = predict_score(self.comment_model, X)

        # 3. Classify reception
        reception_label, confidence, label_distribution = classify_reception_with_confidence(
            predicted_score, predicted_comments
        )

        # 4. Embed title and retrieve similar stories
        query_embedding = embed_texts([title])[0]
        similar_stories = retrieve_similar_stories(query_embedding, self.lancedb_path, top_k=5)

        # 5. Retrieve comments for top similar stories
        similar_comments: list[dict] = []
        for story in similar_stories[:3]:
            story_id = story.get("id")
            if story_id is not None:
                comments = retrieve_comments_for_story(int(story_id), self.lancedb_path, limit=10)
                similar_comments.extend(comments)

        # 6. Generate AI comments if requested
        # client=None triggers Claude CLI fallback in generate_comments()
        simulated_comments: list[dict] = []
        if generate_comments:
            from hn_simulator.comments.generate import generate_comments as _generate_comments

            simulated_comments = _generate_comments(
                title=title,
                description=description,
                predicted_score=predicted_score,
                predicted_label=reception_label,
                similar_stories=similar_stories,
                similar_comments=similar_comments,
                client=self.claude_client,
            )

        # 7. Compute v2 enrichments
        percentile: float | None = None
        expected_score_val: float | None = None
        shap_features: list[dict] = []
        time_recommendation: str = ""

        # Multiclass model: expected score from class probs
        if self.multiclass_model is not None:
            probs = self.multiclass_model.predict(X)  # shape (1, 5)
            expected_score_val = expected_score_from_probs(probs[0])

        # Percentile calibration
        if self.sorted_scores is not None:
            from hn_simulator.model.calibrate import score_to_percentile
            percentile = score_to_percentile(predicted_score, self.sorted_scores)

        # SHAP explanation — use whichever model is available
        try:
            from hn_simulator.model.explain import explain_prediction
            explain_model = self.multiclass_model if self.multiclass_model is not None else self.score_model
            _, feature_names = build_feature_matrix_for_input("", "")
            shap_features = explain_prediction(
                explain_model, X, feature_names, top_k=5
            )
        except Exception:
            shap_features = []

        # Time recommendation
        if self.time_stats is not None:
            from hn_simulator.model.calibrate import recommend_posting_time
            hourly, daily = self.time_stats
            rec = recommend_posting_time(hourly, daily)
            time_recommendation = (
                f"Best posting time: {rec['best_hour']} UTC on {rec['best_day_name']}"
            )

        return SimulationResult(
            predicted_score=predicted_score,
            predicted_comments=predicted_comments,
            reception_label=reception_label,
            confidence=confidence,
            label_distribution=label_distribution,
            simulated_comments=simulated_comments,
            similar_stories=similar_stories,
            percentile=percentile,
            shap_features=shap_features,
            time_recommendation=time_recommendation,
            expected_score=expected_score_val,
        )
