# Beast-Plan Context — HN Simulator v2

## Task Description
8 improvements to the existing HN Reaction Simulator (~/code/hn-simulator):
1. Domain reputation features — avg score per domain as LightGBM feature
2. Percentile calibration — show "top X%" instead of absolute score
3. Time-of-day optimization — recommend best posting time
4. Classification instead of regression — multiclass classifier + percentile-based expected score
5. Better RAG context — include full comment threads from 319K comments
6. Iterative optimization loop — suggest → score → suggest better (max 5 iterations)
7. Backtesting — train on 2023, test on 2024-2025 + calibration plot on known viral posts
8. "Why" explanations — SHAP values per prediction + Claude text summary

## Codebase Summary
- Python 3.11+, uv, hatchling, Click CLI
- 167 tests, 16 source modules
- LightGBM regression (log1p) → 397 features (13 structural + 384 MiniLM embeddings)
- LanceDB for RAG (144K stories, 319K comments indexed)
- Claude CLI runner (headless, no API key) for comment generation
- Current score prediction: Spearman ρ=0.316, range 4-13 (heavily compressed)
- Data: 144K stories (2023-2025), 319K comments, all from open-index/hacker-news

## Decisions
1. **Classification model:** Classification only + percentile-based expected score from bucket medians (replace regression). Buckets: flop(1-3), low(4-15), moderate(16-100), hot(101-300), viral(300+).
2. **Iterative optimization loop:** Max 5 iterations of suggest → score → suggest better.
3. **Backtesting:** Train on 2023, test on 2024-2025 (temporal split) + calibration on known viral posts. Generate calibration plot.
4. **"Why" explanations:** SHAP values per prediction (top-5 features) + Claude text summary combining SHAP with cultural context.

## Scope
### In Scope
- Domain reputation feature (computed from training data, stored as lookup dict)
- Percentile calibration (map predicted bucket → historical percentile)
- Time-of-day analysis (compute hourly/daily score distributions, recommend optimal window)
- Multiclass LightGBM classifier replacing regression
- Enhanced RAG with full comment threads
- Iterative suggest loop (max 5)
- Backtesting with temporal split + calibration report
- SHAP-based feature importance per prediction + Claude explanation
- Updated CLI output format for all new features
- Updated tests for all new functionality

### Out of Scope
- Fine-tuning local LLM (Phase 3)
- Streamlit UI (separate task)
- Author-level features (requires user identity data)
- Downloading pre-2023 data

## Constraints
- M4 Pro 24GB — SHAP computation must be efficient for 397 features
- No API keys needed — Claude CLI for all LLM calls
- Existing 167 tests must continue to pass
- Backward-compatible CLI (existing commands still work)
