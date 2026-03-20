# Research Report — HN Simulator v2

## Key Findings

### 1. Domain Reputation
- 34,436 unique domains in 144K stories. 2,692 with ≥5 posts.
- Use Bayesian smoothing: `(count * domain_mean + k * global_mean) / (count + k)`, k=10
- Global mean=47.57 for unseen domain fallback
- Add 2 features: `domain_avg_score` (smoothed), `domain_post_count`
- Store as JSON dict alongside model (~938KB)

### 2. Percentile Calibration
- `np.searchsorted` on sorted historical scores
- score=9 → top 48%, score=41 → top 25%, score=127 → top 10%, score=529 → top 1%
- Save sorted_scores.npy (~563KB) with model

### 3. Time-of-Day
- Best hours (UTC): 9 (53.2 avg), 3 (51.8), 11 (51.6)
- Best days: Sunday (50.8), Monday (49.2)
- Caveat: UTC times, HN is US-centric. 9 UTC = 5am EST.
- Store hourly_stats + daily_stats dicts with model

### 4. Multiclass Classification
- LightGBM: `objective='multiclass', num_class=5, metric='multi_logloss'`
- 5 buckets: flop(1-3), low(4-15), moderate(16-100), hot(101-300), viral(300+)
- Bucket medians from data: [1.0, 7.0, 37.0, 159.0, 441.0]
- Expected score = probs @ bucket_medians
- Class weights: flop=0.685, low=0.659, moderate=0.717, hot=2.097, viral=6.691
- Must update SCORE_THRESHOLDS and labels.py for 5-class system

### 5. Better RAG Context
- 97.2% comments are top-level (parent = story_id)
- CRITICAL: current retrieve.py does `table.to_pandas()` (270ms full scan) — replace with `.where()` filter (5ms)
- Flat list format is appropriate for 97.2% top-level comments
- Option: store top_comments_text directly in story LanceDB table

### 6. Iterative Optimization Loop
- Pattern: suggest → score → suggest better, track seen_titles to avoid repeats
- Convergence: stop when improvement < min_improvement threshold
- Pass previously_tried titles to Claude prompt
- ~5-10s per Claude CLI call, 5 iterations = 30-90s max

### 7. Backtesting
- CRITICAL: 2023 data has median=2 vs 2024 median=16 — stratified sampling artifact
- Use random 80/20 split, NOT temporal split by year
- Report: accuracy, confusion matrix (5×5), Spearman ρ
- matplotlib NOT installed — use ASCII calibration chart (consistent with CLI style)
- Add matplotlib to pyproject.toml for optional plot generation

### 8. SHAP Explanations
- shap 0.51.0 + LightGBM TreeExplainer: 0.1ms per prediction (regression), 0.3ms (multiclass)
- Regression: shape (1, 397). Multiclass: shape (1, 397, 5)
- For multiclass top-5: mean(|shap_vals|, axis=-1) across classes
- Filter to structural features only for Claude explanation (embedding features not human-readable)
- shap NOT in pyproject.toml — must add

## New Dependencies
- `shap>=0.51.0` — SHAP explanations
- `matplotlib>=3.8.0` — optional, for calibration plots

## Compatibility Warnings
- LanceDB 0.30.0: `table_names()` deprecated → use `list_tables()`
- SHAP multiclass shape: (n, 397, 5) not (5, n, 397)
- LightGBM params: must strip n_estimators/early_stopping_rounds before lgb.train()
- labels.py: current 4-class → must update to 5-class
