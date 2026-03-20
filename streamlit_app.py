"""Streamlit web UI for HN Reaction Simulator."""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COLORS: dict[str, str] = {
    "flop": "#e74c3c",
    "low": "#f39c12",
    "moderate": "#2ecc71",
    "hot": "#3498db",
    "viral": "#9b59b6",
}

VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Helper functions (pure — testable without Streamlit)
# ---------------------------------------------------------------------------

def format_score_display(score: float) -> str:
    """Return a human-readable score string."""
    return f"~{int(score)} points"


@st.cache_resource
def load_simulator(data_dir=None):
    """Load HNSimulator using v2 model paths from config."""
    from hackernews_simulator.config import MODELS_DIR, LANCEDB_DIR, PROCESSED_DIR
    from hackernews_simulator.simulator import HNSimulator

    score_model_path = MODELS_DIR / "score_model.txt"
    comment_model_path = MODELS_DIR / "comment_model.txt"
    multiclass_path = MODELS_DIR / "multiclass_model.txt"
    sorted_scores_path = PROCESSED_DIR / "sorted_scores.npy"
    time_stats_path = PROCESSED_DIR / "time_stats.json"

    if data_dir is not None:
        from pathlib import Path
        base = Path(data_dir)
        score_model_path = base / "models" / "score_model.txt"
        comment_model_path = base / "models" / "comment_model.txt"
        multiclass_path = base / "models" / "multiclass_model.txt"
        sorted_scores_path = base / "processed" / "sorted_scores.npy"
        time_stats_path = base / "processed" / "time_stats.json"

    return HNSimulator(
        score_model_path=score_model_path,
        comment_model_path=comment_model_path,
        lancedb_path=LANCEDB_DIR if data_dir is None else Path(data_dir) / "lancedb",
        multiclass_model_path=multiclass_path if multiclass_path.exists() else None,
        sorted_scores_path=sorted_scores_path if sorted_scores_path.exists() else None,
        time_stats_path=time_stats_path if time_stats_path.exists() else None,
    )


def _colored_label(label: str) -> str:
    """Return an HTML-colored badge for a reception label."""
    color = LABEL_COLORS.get(label, "#888888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{label.upper()}</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("HN Reaction Simulator")
        st.caption("Predict how Hacker News will react to your post before you submit it.")
        st.divider()
        with st.expander("About"):
            st.markdown(f"**Version:** {VERSION}")
            st.markdown(
                "This tool uses machine learning models trained on historical "
                "Hacker News data to predict score, comment count, and community "
                "reception for a given story pitch."
            )
            st.markdown(
                "**Limitations:** Predictions are probabilistic estimates based "
                "on historical patterns. Viral posts are inherently unpredictable. "
                "Comment generation uses Claude AI and requires API access."
            )


# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------

def _render_predict_tab() -> None:
    st.header("Predict HN Reaction")

    title = st.text_input("Story title", placeholder="Show HN: My project does X in Y milliseconds")
    description = st.text_area("Description (optional)", placeholder="Brief summary of your project...")
    generate_comments = st.checkbox("Generate simulated comments", value=True)

    if st.button("Predict", type="primary", key="predict_btn"):
        if not title.strip():
            st.warning("Please enter a title.")
            return

        with st.spinner("Running simulation..."):
            try:
                simulator = load_simulator()
            except Exception as exc:
                st.error(f"Failed to load models: {exc}")
                return

            try:
                result = simulator.simulate(
                    title=title,
                    description=description,
                    generate_comments=generate_comments,
                )
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

        # --- Metrics row ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Score", format_score_display(result.predicted_score))
        col2.metric("Predicted Comments", f"~{int(result.predicted_comments)}")
        label_color = LABEL_COLORS.get(result.reception_label, "#888888")
        col3.markdown(
            f"**Reception**<br>{_colored_label(result.reception_label)} "
            f"<small>({int(result.confidence * 100)}% confidence)</small>",
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Label distribution bar chart ---
        if result.label_distribution:
            st.subheader("Label Distribution")
            import pandas as pd
            dist_df = pd.DataFrame(
                {"label": list(result.label_distribution.keys()),
                 "probability": list(result.label_distribution.values())}
            ).set_index("label")
            st.bar_chart(dist_df)

        # --- Percentile / expected score ---
        info_parts: list[str] = []
        if result.percentile is not None:
            info_parts.append(f"Top **{result.percentile:.1f}%** of HN stories")
        if result.expected_score is not None:
            info_parts.append(f"Expected score (multiclass): ~**{int(result.expected_score)}** points")
        if info_parts:
            st.info("  |  ".join(info_parts))

        # --- SHAP features ---
        if result.shap_features:
            st.subheader("Why This Score")
            import pandas as pd
            shap_df = pd.DataFrame(result.shap_features)
            shap_df["bar"] = shap_df["importance"]
            shap_display = shap_df[["feature", "importance"]].set_index("feature")
            st.bar_chart(shap_display)

        # --- Time recommendation ---
        if result.time_recommendation:
            st.info(f"Posting advice: {result.time_recommendation}")

        # --- Simulated comments ---
        if result.simulated_comments:
            st.subheader("Simulated Comments")
            for comment in result.simulated_comments:
                author = comment.get("username", comment.get("author", "hn_user"))
                text = comment.get("comment", comment.get("text", ""))
                tone = comment.get("tone", "")
                label = f"{author}" + (f" [{tone}]" if tone else "")
                with st.expander(label):
                    st.write(text)


# ---------------------------------------------------------------------------
# Tab 2: Compare
# ---------------------------------------------------------------------------

def _render_compare_tab() -> None:
    st.header("Compare Variants")
    st.caption("Add multiple title/description variants and compare their predicted scores side by side.")

    # Dynamic variant list stored in session state
    if "compare_variants" not in st.session_state:
        st.session_state.compare_variants = [
            {"title": "", "description": ""},
            {"title": "", "description": ""},
        ]

    variants = st.session_state.compare_variants

    for idx, variant in enumerate(variants):
        with st.container():
            st.markdown(f"**Variant {idx + 1}**")
            c1, c2 = st.columns([2, 3])
            with c1:
                variants[idx]["title"] = st.text_input(
                    "Title", value=variant["title"], key=f"cmp_title_{idx}",
                    placeholder="Show HN: ..."
                )
            with c2:
                variants[idx]["description"] = st.text_input(
                    "Description", value=variant["description"], key=f"cmp_desc_{idx}",
                    placeholder="Optional description"
                )

    col_add, col_compare = st.columns([1, 3])
    with col_add:
        if st.button("Add variant"):
            st.session_state.compare_variants.append({"title": "", "description": ""})
            st.rerun()

    with col_compare:
        run_compare = st.button("Compare", type="primary", key="compare_btn")

    if run_compare:
        filled = [v for v in variants if v["title"].strip()]
        if len(filled) < 2:
            st.warning("Please enter at least 2 titles to compare.")
            return

        with st.spinner("Scoring all variants..."):
            try:
                simulator = load_simulator()
            except Exception as exc:
                st.error(f"Failed to load models: {exc}")
                return

            rows = []
            for v in filled:
                try:
                    result = simulator.simulate(
                        title=v["title"],
                        description=v["description"],
                        generate_comments=False,
                    )
                    rows.append({
                        "Title": v["title"],
                        "Score": int(result.predicted_score),
                        "Comments": int(result.predicted_comments),
                        "Reception": result.reception_label,
                        "Confidence": f"{int(result.confidence * 100)}%",
                    })
                except Exception as exc:
                    rows.append({
                        "Title": v["title"],
                        "Score": 0,
                        "Comments": 0,
                        "Reception": "error",
                        "Confidence": "-",
                    })

        import pandas as pd
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

        # Color the Reception column
        def color_reception(val: str) -> str:
            c = LABEL_COLORS.get(val, "#888888")
            return f"background-color: {c}; color: white"

        st.dataframe(
            df.style.applymap(color_reception, subset=["Reception"]),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: Suggest Loop
# ---------------------------------------------------------------------------

def _render_suggest_tab() -> None:
    st.header("Suggest Loop — Optimize Your Title")
    st.caption("Iteratively generate and score alternative titles to find the best-performing variant.")

    title = st.text_input("Original title", placeholder="My project does X", key="suggest_title")
    description = st.text_area("Description (optional)", placeholder="What it does...", key="suggest_desc")
    max_iterations = st.slider("Max iterations", min_value=1, max_value=10, value=5)

    if st.button("Optimize", type="primary", key="suggest_btn"):
        if not title.strip():
            st.warning("Please enter a title.")
            return

        progress_bar = st.progress(0, text="Initializing...")

        with st.spinner("Running suggest loop..."):
            try:
                simulator = load_simulator()
            except Exception as exc:
                st.error(f"Failed to load models: {exc}")
                return

            from hackernews_simulator.suggest import iterative_optimize

            # We wrap iterative_optimize and update progress per iteration
            # Since iterative_optimize is synchronous we simulate progress steps
            progress_bar.progress(10, text="Running iteration 1...")

            try:
                result = iterative_optimize(
                    simulator=simulator,
                    original={"title": title, "description": description},
                    client=None,
                    max_iterations=max_iterations,
                )
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")
                return

            progress_bar.progress(100, text="Done.")

        best = result["best"]
        all_variants = result["all_variants"]
        iterations = result["iterations"]
        improvement = result["improvement"]

        st.success(
            f"Completed {iterations} iteration(s). "
            f"Score improvement: +{improvement:.1f} points."
        )

        # Best title highlighted
        st.subheader("Best Title")
        label_color = "#2ecc71" if improvement > 0 else "#95a5a6"
        st.markdown(
            f'<div style="border-left:4px solid {label_color};padding:8px 12px;'
            f'background:#f9f9f9;border-radius:4px">'
            f'<b>{best["title"]}</b><br>'
            f'<small>Predicted score: {format_score_display(best["predicted_score"])}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # All variants table
        if all_variants:
            st.subheader("All Variants Tried")
            import pandas as pd
            variants_df = pd.DataFrame([
                {
                    "Title": v["title"],
                    "Predicted Score": int(v["predicted_score"]),
                }
                for v in all_variants
            ]).sort_values("Predicted Score", ascending=False).reset_index(drop=True)
            st.dataframe(variants_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="HN Reaction Simulator",
        page_icon=":orange_circle:",
        layout="wide",
    )

    _render_sidebar()

    tab_predict, tab_compare, tab_suggest = st.tabs(["Predict", "Compare", "Suggest Loop"])

    with tab_predict:
        _render_predict_tab()

    with tab_compare:
        _render_compare_tab()

    with tab_suggest:
        _render_suggest_tab()


if __name__ == "__main__":
    main()
