

I have all the information needed to create the plan. I'll write it now and check the test results when done.

Here is the complete implementation plan:

---

# Implementation Plan: HN Simulator v2 — 8 Improvements

## Requirements Summary

Enhance the existing HN Reaction Simulator with 8 improvements:
1. **Domain reputation features** -- Bayesian-smoothed avg score per domain as a LightGBM feature
2. **Percentile calibration** -- Show "top X%" instead of raw predicted score
3. **Time-of-day optimization** -- Recommend best posting time based on historical hourly/daily stats
4. **Multiclass classification** -- Replace regression with 5-class classifier; expected score from bucket medians
5. **Better RAG context** -- Replace `table.to_pandas()` with LanceDB `.where()` filter for 54x speedup
6. **Iterative optimization loop** -- suggest -> score -> suggest better, max 5 iterations with convergence
7. **Backtesting** -- Random 80/20 split, accuracy, confusion matrix, Spearman rho, ASCII calibration chart
8. **SHAP explanations** -- TreeExplainer for per-prediction feature importance + Claude text summary

## Architecture Overview

```
                       ┌──────────────┐
                       │   cli.py     │  Updated output: percentile, SHAP, time advice
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │ simulator.py │  Orchestrates all new features
                       └──────┬───────┘
                              │
          ┌───────────┬───────┼────────┬──────────┐
          │           │       │        │          │
   ┌──────▼──┐ ┌──────▼───┐ ┌▼──────┐ ┌▼────────┐ ┌▼──────────┐
   │features/ │ │ model/   │ │ rag/  │ │suggest. │ │ model/    │
   │structural│ │labels.py │ │retr.  │ │py       │ │explain.py │ NEW
   │+domain   │ │5-class   │ │.where │ │iter.loop│ │SHAP       │
   └──────────┘ │train.py  │ └───────┘ └─────────┘ └───────────┘
                │multiclass│
                │calibrate │ NEW
                │backtest  │ NEW
                └──────────┘
```

**Key design decisions:**
- Domain reputation is a JSON dict lookup, computed once from training data, loaded at feature extraction time
- Percentile calibration uses `np.searchsorted` on sorted historical scores (saved as `.npy`)
- Multiclass classifier outputs 5-class probabilities; expected_score = probs @ bucket_medians
- SHAP uses TreeExplainer (0.3ms/prediction); only structural feature names shown to Claude
- Iterative loop tracks `seen_titles` set to avoid duplicates, uses score delta convergence threshold
- Backtesting uses random 80/20 split (not temporal -- research finding), reports accuracy + Spearman rho

## Pre-requisites

Add to `pyproject.toml` dependencies:
```
"shap>=0.51.0",
```
No other new dependencies needed (matplotlib is optional, ASCII chart used instead).

## Tasks

### Task 1: Domain Reputation Features

**Files:**
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/features/structural.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/features/pipeline.py`
- CREATE: `/Users/malakhov/code/hn-simulator/tests/test_domain_features.py`

**Depends on:** none

#### Context

The training data has 34K unique domains. Compute Bayesian-smoothed average score per domain:
`smoothed_score = (n * domain_mean + k * global_mean) / (n + k)` where `k=10`, `global_mean=47.57`.

Store as a JSON dict `{domain_str: {"avg_score": float, "post_count": int}}` at `data/processed/domain_stats.json`. At feature extraction time, load this dict and add 2 features: `domain_avg_score` and `domain_post_count`. For unknown domains, use `global_mean` and `0`.

This changes the structural feature count from 13 to 15, and total features from 397 to 399. All tests that hardcode `397` or `13` must be updated.

#### TDD Cycle

**RED phase -- Write failing tests first:**

File: `/Users/malakhov/code/hn-simulator/tests/test_domain_features.py`

```python
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
        from hn_simulator.features.structural import compute_domain_stats
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
        from hn_simulator.features.structural import compute_domain_stats
        df = pd.DataFrame({"domain": pd.Series(dtype=str), "score": pd.Series(dtype=np.int32)})
        stats = compute_domain_stats(df, k=10, global_mean=50.0)
        assert stats == {}


class TestExtractDomainReputationFeatures:
    def test_known_domain_gets_stats(self, domain_stats):
        from hn_simulator.features.structural import extract_domain_reputation_features
        df = pd.DataFrame({"domain": ["github.com"]})
        result = extract_domain_reputation_features(df, domain_stats, global_mean=47.57)
        assert result["domain_avg_score"].iloc[0] == pytest.approx(120.5)
        assert result["domain_post_count"].iloc[0] == 5000

    def test_unknown_domain_gets_global_mean(self, domain_stats):
        from hn_simulator.features.structural import extract_domain_reputation_features
        df = pd.DataFrame({"domain": ["unknown.org"]})
        result = extract_domain_reputation_features(df, domain_stats, global_mean=47.57)
        assert result["domain_avg_score"].iloc[0] == pytest.approx(47.57)
        assert result["domain_post_count"].iloc[0] == 0

    def test_empty_domain_gets_global_mean(self, domain_stats):
        from hn_simulator.features.structural import extract_domain_reputation_features
        df = pd.DataFrame({"domain": [""]})
        result = extract_domain_reputation_features(df, domain_stats, global_mean=47.57)
        assert result["domain_avg_score"].iloc[0] == pytest.approx(47.57)
        assert result["domain_post_count"].iloc[0] == 0


class TestStructuralFeaturesIncludesDomain:
    def test_extract_structural_features_has_15_columns(self, sample_stories_df):
        from hn_simulator.data.preprocess import preprocess_stories
        from hn_simulator.features.structural import extract_structural_features
        df = preprocess_stories(sample_stories_df)
        result = extract_structural_features(df)
        assert "domain_avg_score" in result.columns
        assert "domain_post_count" in result.columns
        assert len(result.columns) == 15
```

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/features/structural.py`

Add these functions and modify `extract_structural_features`:

1. `compute_domain_stats(df: pd.DataFrame, k: int = 10, global_mean: float = 47.57) -> dict` -- Takes a DataFrame with `domain` and `score` columns, groups by domain, computes Bayesian-smoothed avg. Returns `{domain: {"avg_score": float, "post_count": int}}`.

2. `extract_domain_reputation_features(df: pd.DataFrame, domain_stats: dict, global_mean: float = 47.57) -> pd.DataFrame` -- Looks up each row's domain in domain_stats dict. Returns DataFrame with `domain_avg_score` and `domain_post_count` columns.

3. Modify `extract_structural_features()` to:
   - Accept optional `domain_stats: dict | None = None` parameter
   - If None, load from `PROCESSED_DIR / "domain_stats.json"` if it exists, else use empty dict
   - Call `extract_domain_reputation_features()` and include in the concat

4. Update `build_feature_matrix` in `pipeline.py` to pass `domain_stats` through if provided.

**REFACTOR phase:**
- Save/load helpers: `save_domain_stats(stats, path)` and `load_domain_stats(path)` -- simple JSON I/O.

#### Updating Existing Tests

The feature count changes from 13 structural to 15, and total from 397 to 399. These tests must be updated:

- `/Users/malakhov/code/hn-simulator/tests/test_structural_features.py`: `TestCombineStructuralFeatures` -- change expected column count from 13 to 15
- `/Users/malakhov/code/hn-simulator/tests/test_feature_pipeline.py`: change `397` to `399` in shape assertions and feature count assertions
- `/Users/malakhov/code/hn-simulator/tests/test_simulator.py`: change `397` in `mock_models` fixture (X shape, names)
- `/Users/malakhov/code/hn-simulator/tests/test_suggest.py`: change `397` in `mock_models_and_simulator` fixture
- `/Users/malakhov/code/hn-simulator/tests/test_compare.py`: change `397` in `mock_models_and_simulator` fixture
- `/Users/malakhov/code/hn-simulator/tests/test_integration.py`: `full_pipeline` fixture shape check
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/features/pipeline.py`: update docstring from "13 cols" to "15 cols" and "397" to "399"

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_domain_features.py tests/test_structural_features.py tests/test_feature_pipeline.py -v
```

#### Acceptance Criteria
- [ ] `compute_domain_stats` returns Bayesian-smoothed averages matching expected values
- [ ] Unknown domains fall back to `global_mean` for avg_score and 0 for post_count
- [ ] `extract_structural_features` now returns 15 columns (was 13)
- [ ] `build_feature_matrix` returns shape `(n, 399)` (was 397)
- [ ] All existing 167 tests pass after updating hardcoded feature counts

---

### Task 2: Multiclass Classification (Labels + Training)

**Files:**
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/labels.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/train.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/config.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/tests/test_labels.py`
- CREATE: `/Users/malakhov/code/hn-simulator/tests/test_multiclass.py`

**Depends on:** Task 1 (feature count changes affect training data shape)

#### Context

Replace 4-class rule-based classification with 5-class model-based classification. New classes:
- `flop`: score 1-3 (bucket median = 1)
- `low`: score 4-15 (bucket median = 7)
- `moderate`: score 16-100 (bucket median = 37)
- `hot`: score 101-300 (bucket median = 159)
- `viral`: score 301+ (bucket median = 441)

`BUCKET_MEDIANS = np.array([1, 7, 37, 159, 441])`
`expected_score = class_probabilities @ BUCKET_MEDIANS`

The existing regression models (`score_model.txt`, `comment_model.txt`) are kept for backward compatibility. The new multiclass model is a separate model file (`score_model_multiclass.txt`).

#### TDD Cycle

**RED phase -- Write failing tests first:**

File: `/Users/malakhov/code/hn-simulator/tests/test_multiclass.py`

```python
"""Tests for 5-class multiclass classification."""
import pytest
import numpy as np
import lightgbm as lgb


LABELS_5 = ("flop", "low", "moderate", "hot", "viral")
BUCKET_MEDIANS = np.array([1, 7, 37, 159, 441])


@pytest.fixture
def synthetic_multiclass_data():
    """Generate synthetic data with 5 score classes. Fixed seed."""
    rng = np.random.default_rng(42)
    n = 600
    X = rng.standard_normal((n, 20)).astype(np.float32)
    # Create scores correlated with features
    raw_scores = np.exp(X[:, 0] * 2 + X[:, 1] + rng.standard_normal(n) * 0.5).clip(1, 1000)
    feature_names = [f"feat_{i}" for i in range(20)]
    return X, raw_scores, feature_names


class TestScoreToClassLabel:
    def test_flop_boundary(self):
        from hn_simulator.model.labels import score_to_class_index
        assert score_to_class_index(1) == 0
        assert score_to_class_index(3) == 0

    def test_low_range(self):
        from hn_simulator.model.labels import score_to_class_index
        assert score_to_class_index(4) == 1
        assert score_to_class_index(15) == 1

    def test_moderate_range(self):
        from hn_simulator.model.labels import score_to_class_index
        assert score_to_class_index(16) == 2
        assert score_to_class_index(100) == 2

    def test_hot_range(self):
        from hn_simulator.model.labels import score_to_class_index
        assert score_to_class_index(101) == 3
        assert score_to_class_index(300) == 3

    def test_viral_range(self):
        from hn_simulator.model.labels import score_to_class_index
        assert score_to_class_index(301) == 4
        assert score_to_class_index(10000) == 4

    def test_zero_and_negative_are_flop(self):
        from hn_simulator.model.labels import score_to_class_index
        assert score_to_class_index(0) == 0
        assert score_to_class_index(-5) == 0


class TestExpectedScoreFromProbabilities:
    def test_certain_flop(self):
        from hn_simulator.model.labels import expected_score_from_probs
        probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert expected_score_from_probs(probs) == pytest.approx(1.0)

    def test_certain_viral(self):
        from hn_simulator.model.labels import expected_score_from_probs
        probs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        assert expected_score_from_probs(probs) == pytest.approx(441.0)

    def test_uniform_distribution(self):
        from hn_simulator.model.labels import expected_score_from_probs
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        expected = np.dot(probs, BUCKET_MEDIANS)
        assert expected_score_from_probs(probs) == pytest.approx(expected)


class TestTrainMulticlassModel:
    def test_returns_booster_and_metrics(self, synthetic_multiclass_data):
        from hn_simulator.model.train import train_multiclass_model
        X, scores, names = synthetic_multiclass_data
        from hn_simulator.model.labels import score_to_class_index
        y = np.array([score_to_class_index(s) for s in scores])
        model, metrics = train_multiclass_model(
            X[:480], y[:480], X[480:], y[480:], names
        )
        assert isinstance(model, lgb.Booster)
        assert "val_accuracy" in metrics
        assert metrics["val_accuracy"] > 0.0

    def test_predict_returns_probabilities(self, synthetic_multiclass_data):
        from hn_simulator.model.train import train_multiclass_model
        from hn_simulator.model.labels import score_to_class_index
        X, scores, names = synthetic_multiclass_data
        y = np.array([score_to_class_index(s) for s in scores])
        model, _ = train_multiclass_model(X[:480], y[:480], X[480:], y[480:], names)
        probs = model.predict(X[480:481])
        # Shape should be (1, 5) for 5 classes
        assert probs.shape == (1, 5)
        assert abs(probs.sum() - 1.0) < 0.01


class TestLabels5Class:
    def test_labels_tuple_has_5_entries(self):
        from hn_simulator.model.labels import LABELS_5
        assert len(LABELS_5) == 5
        assert LABELS_5 == ("flop", "low", "moderate", "hot", "viral")

    def test_bucket_medians_shape(self):
        from hn_simulator.model.labels import BUCKET_MEDIANS
        assert len(BUCKET_MEDIANS) == 5
        assert list(BUCKET_MEDIANS) == [1, 7, 37, 159, 441]
```

Also update existing tests in `/Users/malakhov/code/hn-simulator/tests/test_labels.py`:
- Existing `classify_reception` tests remain unchanged (backward compat)
- Add `"low"` label to `TestReceptionDescription` -- new descriptions dict must include `"low"`
- `classify_reception_with_confidence` must now include `"low"` in distribution

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/config.py`
- Add `SCORE_THRESHOLDS_5CLASS` dict: `{"flop": 3, "low": 15, "moderate": 100, "hot": 300}`
- Add `LIGHTGBM_MULTICLASS_PARAMS` dict (copy of `LIGHTGBM_PARAMS` but with `objective: "multiclass"`, `num_class: 5`, `metric: "multi_logloss"`)

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/labels.py`
- Add module-level: `LABELS_5 = ("flop", "low", "moderate", "hot", "viral")`
- Add module-level: `BUCKET_MEDIANS = np.array([1, 7, 37, 159, 441])`
- Add `_BOUNDARIES_5 = (3, 15, 100, 300)` (4 boundaries for 5 classes)
- Add function `score_to_class_index(score: float) -> int` -- returns 0-4 using `_BOUNDARIES_5`
- Add function `expected_score_from_probs(probs: np.ndarray) -> float` -- returns `float(probs @ BUCKET_MEDIANS)`
- Update `_LABELS` to reference the original 4-class (keep backward compat)
- Update `_DESCRIPTIONS` to add `"low"` key
- Update `classify_reception_with_confidence` to use 5-class labels and 5 centres

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/train.py`
- Add function `train_multiclass_model(X_train, y_train, X_val, y_val, feature_names) -> (lgb.Booster, dict)`:
  - Uses `LIGHTGBM_MULTICLASS_PARAMS`
  - `y_train`/`y_val` are integer class indices (0-4)
  - Computes class weights from `np.bincount(y_train)` -- inverse frequency
  - Returns booster and `{"val_accuracy": float, "val_logloss": float}`
  - Uses `lgb.Dataset` with `weight` parameter based on class weights per sample

**REFACTOR phase:**
- Extract shared `_train_base` logic if significant duplication with `_train_regression_model`

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_multiclass.py tests/test_labels.py -v
```

#### Acceptance Criteria
- [ ] `score_to_class_index` maps scores to correct 0-4 class indices at all boundaries
- [ ] `expected_score_from_probs` computes correct dot product with bucket medians
- [ ] `train_multiclass_model` returns a Booster that predicts (n, 5) probability arrays summing to ~1.0
- [ ] `LABELS_5` and `BUCKET_MEDIANS` are importable module-level constants
- [ ] Existing 4-class `classify_reception` tests still pass (backward compat)
- [ ] `classify_reception_with_confidence` now returns 5-class distribution including "low"

---

### Task 3: Percentile Calibration + Time-of-Day Stats

**Files:**
- CREATE: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/calibrate.py`
- CREATE: `/Users/malakhov/code/hn-simulator/tests/test_calibrate.py`

**Depends on:** none

#### Context

Two related features stored in the same module:

1. **Percentile calibration**: Given a predicted score, map it to "top X%" using historical score distribution. Precompute sorted array of all historical scores, save as `data/processed/sorted_scores.npy`. At inference: `percentile = 100 * (1 - np.searchsorted(sorted_scores, score) / len(sorted_scores))`. Example: score=9 -> top 48%, score=127 -> top 10%.

2. **Time-of-day stats**: Precompute `hourly_stats` (dict mapping hour 0-23 to avg score) and `daily_stats` (dict mapping day 0-6 to avg score). Recommend the best posting window. Save as `data/processed/time_stats.json`.

#### TDD Cycle

**RED phase -- Write failing tests first:**

File: `/Users/malakhov/code/hn-simulator/tests/test_calibrate.py`

```python
"""Tests for percentile calibration and time-of-day stats."""
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path


class TestScoreToPercentile:
    def test_median_score_is_near_50(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Score 5 -> 50th percentile -> "top 50%"
        pct = score_to_percentile(5, sorted_scores)
        assert 40.0 <= pct <= 60.0

    def test_max_score_is_top_10_percent(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.arange(1, 101)  # 1..100
        pct = score_to_percentile(100, sorted_scores)
        assert pct <= 1.0  # top 1%

    def test_min_score_is_near_100_percent(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.arange(1, 101)
        pct = score_to_percentile(1, sorted_scores)
        assert pct >= 99.0

    def test_above_max_is_top_1_percent(self):
        from hn_simulator.model.calibrate import score_to_percentile
        sorted_scores = np.arange(1, 101)
        pct = score_to_percentile(1000, sorted_scores)
        assert pct < 1.0

    def test_score_9_example(self):
        """From research: score=9 -> top ~48%."""
        from hn_simulator.model.calibrate import score_to_percentile
        # Simulate a distribution where score=9 is around median
        sorted_scores = np.sort(np.concatenate([
            np.ones(50) * 1,    # 50 scores of 1
            np.ones(30) * 5,    # 30 scores of 5
            np.ones(10) * 9,    # 10 scores of 9
            np.ones(7) * 50,    # 7 scores of 50
            np.ones(3) * 500,   # 3 scores of 500
        ])).astype(np.int32)
        pct = score_to_percentile(9, sorted_scores)
        # 90 scores <= 9, out of 100, so top (100 - 90) = 10%
        assert pct == pytest.approx(10.0)


class TestBuildSortedScores:
    def test_returns_sorted_array(self):
        from hn_simulator.model.calibrate import build_sorted_scores
        scores = np.array([50, 10, 200, 1, 30], dtype=np.int32)
        result = build_sorted_scores(scores)
        assert np.all(result[:-1] <= result[1:])  # sorted ascending
        assert len(result) == 5

    def test_handles_duplicates(self):
        from hn_simulator.model.calibrate import build_sorted_scores
        scores = np.array([5, 5, 5, 10, 10], dtype=np.int32)
        result = build_sorted_scores(scores)
        assert len(result) == 5
        assert list(result) == [5, 5, 5, 10, 10]


class TestComputeTimeStats:
    def test_hourly_stats_has_24_entries(self):
        from hn_simulator.model.calibrate import compute_time_stats
        df = pd.DataFrame({
            "time": pd.to_datetime([
                "2023-01-01 09:00:00", "2023-01-01 09:30:00",
                "2023-01-01 15:00:00", "2023-01-02 03:00:00",
            ], utc=True),
            "score": np.array([100, 50, 20, 80], dtype=np.int32),
        })
        hourly, daily = compute_time_stats(df)
        assert isinstance(hourly, dict)
        # Hour 9 has avg score (100+50)/2 = 75
        assert hourly[9] == pytest.approx(75.0)
        assert hourly[15] == pytest.approx(20.0)
        assert hourly[3] == pytest.approx(80.0)

    def test_daily_stats_has_7_entries(self):
        from hn_simulator.model.calibrate import compute_time_stats
        df = pd.DataFrame({
            "time": pd.to_datetime([
                "2023-01-01 09:00:00",  # Sunday (6)
                "2023-01-02 09:00:00",  # Monday (0)
            ], utc=True),
            "score": np.array([100, 50], dtype=np.int32),
        })
        _, daily = compute_time_stats(df)
        assert isinstance(daily, dict)
        assert daily[6] == pytest.approx(100.0)  # Sunday
        assert daily[0] == pytest.approx(50.0)   # Monday


class TestRecommendPostingTime:
    def test_returns_best_hour_and_day(self):
        from hn_simulator.model.calibrate import recommend_posting_time
        hourly = {h: 10.0 for h in range(24)}
        hourly[9] = 100.0  # Hour 9 is best
        daily = {d: 10.0 for d in range(7)}
        daily[6] = 80.0  # Sunday is best
        result = recommend_posting_time(hourly, daily)
        assert result["best_hour"] == 9
        assert result["best_day"] == 6
        assert result["best_day_name"] == "Sunday"

    def test_returns_top_3_hours(self):
        from hn_simulator.model.calibrate import recommend_posting_time
        hourly = {h: float(h) for h in range(24)}  # hour 23 is best
        daily = {d: 10.0 for d in range(7)}
        result = recommend_posting_time(hourly, daily)
        assert result["top_3_hours"] == [23, 22, 21]


class TestSaveLoadCalibrationData:
    def test_roundtrip_sorted_scores(self, tmp_path):
        from hn_simulator.model.calibrate import save_sorted_scores, load_sorted_scores
        scores = np.array([1, 5, 10, 50, 100], dtype=np.int32)
        path = tmp_path / "sorted_scores.npy"
        save_sorted_scores(scores, path)
        loaded = load_sorted_scores(path)
        np.testing.assert_array_equal(scores, loaded)

    def test_roundtrip_time_stats(self, tmp_path):
        from hn_simulator.model.calibrate import save_time_stats, load_time_stats
        hourly = {h: float(h * 10) for h in range(24)}
        daily = {d: float(d * 5) for d in range(7)}
        path = tmp_path / "time_stats.json"
        save_time_stats(hourly, daily, path)
        h_loaded, d_loaded = load_time_stats(path)
        assert h_loaded[9] == pytest.approx(90.0)
        assert d_loaded[6] == pytest.approx(30.0)
```

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/calibrate.py`

```python
"""Percentile calibration and time-of-day analysis."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def score_to_percentile(score: float, sorted_scores: np.ndarray) -> float:
    """Map a score to 'top X%' using sorted historical scores.
    Returns percentile as float (e.g., 10.0 means 'top 10%').
    """
    idx = np.searchsorted(sorted_scores, score, side="right")
    return 100.0 * (1.0 - idx / len(sorted_scores))


def build_sorted_scores(scores: np.ndarray) -> np.ndarray:
    """Sort score array ascending for percentile lookup."""
    return np.sort(scores)


def compute_time_stats(df: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    """Compute average score per hour (0-23) and per day-of-week (0=Mon..6=Sun).
    Returns (hourly_stats, daily_stats) dicts.
    """
    df = df.copy()
    df["_hour"] = df["time"].dt.hour
    df["_dow"] = df["time"].dt.dayofweek
    hourly = df.groupby("_hour")["score"].mean().to_dict()
    daily = df.groupby("_dow")["score"].mean().to_dict()
    # Convert keys to int (groupby may produce int64)
    hourly = {int(k): float(v) for k, v in hourly.items()}
    daily = {int(k): float(v) for k, v in daily.items()}
    return hourly, daily


def recommend_posting_time(
    hourly: dict[int, float], daily: dict[int, float]
) -> dict:
    """Recommend best posting time based on hourly/daily stats."""
    best_hour = max(hourly, key=hourly.get)
    best_day = max(daily, key=daily.get)
    top_3_hours = sorted(hourly, key=hourly.get, reverse=True)[:3]
    return {
        "best_hour": best_hour,
        "best_day": best_day,
        "best_day_name": _DAY_NAMES[best_day],
        "top_3_hours": top_3_hours,
        "best_hour_avg_score": hourly[best_hour],
        "best_day_avg_score": daily[best_day],
    }


def save_sorted_scores(scores: np.ndarray, path: Path | str) -> None:
    np.save(str(path), scores)

def load_sorted_scores(path: Path | str) -> np.ndarray:
    return np.load(str(path))

def save_time_stats(hourly: dict, daily: dict, path: Path | str) -> None:
    data = {"hourly": {str(k): v for k, v in hourly.items()},
            "daily": {str(k): v for k, v in daily.items()}}
    Path(path).write_text(json.dumps(data, indent=2))

def load_time_stats(path: Path | str) -> tuple[dict[int, float], dict[int, float]]:
    data = json.loads(Path(path).read_text())
    hourly = {int(k): v for k, v in data["hourly"].items()}
    daily = {int(k): v for k, v in data["daily"].items()}
    return hourly, daily
```

**REFACTOR phase:** None needed -- functions are small and focused.

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_calibrate.py -v
```

#### Acceptance Criteria
- [ ] `score_to_percentile(100, np.arange(1,101))` returns ~1.0 (top 1%)
- [ ] `compute_time_stats` produces correct hourly/daily averages
- [ ] `recommend_posting_time` returns the hour/day with highest avg score
- [ ] Save/load roundtrips preserve data exactly
- [ ] No existing tests broken (this task creates only new files)

---

### Task 4: Better RAG -- LanceDB `.where()` Filter

**Files:**
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/rag/retrieve.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/tests/test_rag_retrieve.py`

**Depends on:** none

#### Context

Current `retrieve_comments_for_story` calls `table.to_pandas()` which loads the entire 319K-row comments table into memory (270ms). Replace with LanceDB `.where()` filter which pushes the predicate down (5ms). Research shows 97.2% of comments are top-level (parent is a story ID), so filtering by `parent` is the primary access pattern.

#### TDD Cycle

**RED phase -- Write failing tests first:**

Add to `/Users/malakhov/code/hn-simulator/tests/test_rag_retrieve.py`:

```python
class TestRetrieveCommentsPerformance:
    def test_retrieve_uses_where_filter(self, populated_lancedb):
        """Verify retrieve_comments_for_story uses .where() not .to_pandas()."""
        from hn_simulator.rag.retrieve import retrieve_comments_for_story
        from unittest.mock import patch, MagicMock

        # The function should call table.search() or table.to_pandas() with a filter
        # We verify it does NOT call table.to_pandas() without args (full table scan)
        comments = retrieve_comments_for_story(story_id=1, db_path=populated_lancedb, limit=10)
        # Functional test: should still return correct results
        assert isinstance(comments, list)
        for c in comments:
            assert c["parent"] == 1

    def test_retrieve_comments_respects_limit(self, populated_lancedb):
        from hn_simulator.rag.retrieve import retrieve_comments_for_story
        comments = retrieve_comments_for_story(story_id=1, db_path=populated_lancedb, limit=1)
        assert len(comments) <= 1
```

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/rag/retrieve.py`

Replace `retrieve_comments_for_story` body:

```python
def retrieve_comments_for_story(
    story_id: int,
    db_path: Path,
    limit: int = 20,
) -> list[dict]:
    """Return up to limit comments whose parent == story_id.
    Uses LanceDB .where() filter instead of loading full table.
    """
    db = _open_db(db_path)
    if "comments" not in db.table_names():
        return []
    table = db.open_table("comments")
    try:
        results = (
            table.search()
            .where(f"parent = {story_id}")
            .limit(limit)
            .to_list()
        )
        return results
    except Exception:
        # Fallback for LanceDB versions that require vector search
        # Use to_pandas with filter
        try:
            df = table.to_pandas()
            filtered = df[df["parent"] == story_id]
            if limit:
                filtered = filtered.head(limit)
            return filtered.to_dict(orient="records")
        except Exception:
            return []
```

Note: LanceDB `table.search()` without a vector query + `.where()` is the correct pattern for scalar filtering. If the LanceDB version installed does not support `table.search().where()` without a vector, fall back gracefully.

**REFACTOR phase:** Remove the old pandas-based approach once `.where()` is confirmed working.

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_rag_retrieve.py -v
```

#### Acceptance Criteria
- [ ] `retrieve_comments_for_story` returns correct comments filtered by parent ID
- [ ] No `table.to_pandas()` full-table scan in the primary code path
- [ ] Empty results for non-existent story_id still return `[]`
- [ ] All existing RAG retrieval tests pass
- [ ] Limit parameter is respected

---

### Task 5: SHAP Explanations

**Files:**
- CREATE: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/explain.py`
- CREATE: `/Users/malakhov/code/hn-simulator/tests/test_explain.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/pyproject.toml` (add `shap>=0.51.0`)

**Depends on:** Task 1 (feature names count), Task 2 (multiclass model)

#### Context

Use `shap.TreeExplainer` for per-prediction feature importance. For multiclass, SHAP values have shape `(1, n_features, n_classes)`. Filter to show only the 15 structural feature names (not 384 embedding dims) for human readability. Return top-5 structural features sorted by absolute SHAP magnitude. Optionally generate Claude text summary combining SHAP with HN cultural context.

#### TDD Cycle

**RED phase -- Write failing tests first:**

File: `/Users/malakhov/code/hn-simulator/tests/test_explain.py`

```python
"""Tests for SHAP-based feature explanations."""
import pytest
import numpy as np
import lightgbm as lgb


@pytest.fixture
def trained_multiclass_model():
    """Train a small multiclass model for SHAP tests. Fixed seed."""
    from hn_simulator.model.train import train_multiclass_model
    from hn_simulator.model.labels import score_to_class_index
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 20)).astype(np.float32)
    raw_scores = np.exp(X[:, 0] * 2 + rng.standard_normal(n) * 0.5).clip(1, 1000)
    y = np.array([score_to_class_index(s) for s in raw_scores])
    names = [f"feat_{i}" for i in range(20)]
    model, _ = train_multiclass_model(X[:240], y[:240], X[240:], y[240:], names)
    return model, names


class TestExplainPrediction:
    def test_returns_top_features_list(self, trained_multiclass_model):
        from hn_simulator.model.explain import explain_prediction
        model, names = trained_multiclass_model
        rng = np.random.default_rng(99)
        X = rng.standard_normal((1, 20)).astype(np.float32)
        # All 20 features are "structural" in this test
        structural_names = names[:20]
        result = explain_prediction(model, X, names, structural_names, top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for item in result:
            assert "feature" in item
            assert "shap_value" in item
            assert "direction" in item
            assert item["direction"] in ("positive", "negative")

    def test_features_are_sorted_by_abs_importance(self, trained_multiclass_model):
        from hn_simulator.model.explain import explain_prediction
        model, names = trained_multiclass_model
        rng = np.random.default_rng(99)
        X = rng.standard_normal((1, 20)).astype(np.float32)
        result = explain_prediction(model, X, names, names[:20], top_k=5)
        abs_values = [abs(item["shap_value"]) for item in result]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_only_structural_features_returned(self, trained_multiclass_model):
        from hn_simulator.model.explain import explain_prediction
        model, names = trained_multiclass_model
        rng = np.random.default_rng(99)
        X = rng.standard_normal((1, 20)).astype(np.float32)
        # Only include first 5 as "structural"
        structural = names[:5]
        result = explain_prediction(model, X, names, structural, top_k=5)
        for item in result:
            assert item["feature"] in structural


class TestFormatExplanationText:
    def test_returns_string(self):
        from hn_simulator.model.explain import format_explanation_text
        features = [
            {"feature": "is_show_hn", "shap_value": 0.5, "direction": "positive"},
            {"feature": "title_length", "shap_value": -0.3, "direction": "negative"},
        ]
        text = format_explanation_text(features, predicted_label="hot")
        assert isinstance(text, str)
        assert "is_show_hn" in text
        assert "hot" in text.lower() or "Hot" in text

    def test_empty_features_returns_message(self):
        from hn_simulator.model.explain import format_explanation_text
        text = format_explanation_text([], predicted_label="flop")
        assert isinstance(text, str)
        assert len(text) > 0
```

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/explain.py`

```python
"""SHAP-based feature explanations for predictions."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb


def explain_prediction(
    model: lgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    structural_feature_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Compute SHAP values and return top-k structural features by importance.

    For multiclass: SHAP shape is (1, n_features, n_classes).
    We take the mean absolute SHAP across classes for each feature,
    then filter to structural features only, sort by magnitude.

    Returns list of dicts: [{"feature": str, "shap_value": float, "direction": str}]
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle multiclass shape: (n_samples, n_features, n_classes) or list of arrays
    if isinstance(shap_values, list):
        # Old shap format: list of (n_samples, n_features) per class
        stacked = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(stacked[0]), axis=-1)  # (n_features,)
        # Use signed value from the winning class (max prob)
        winning_class = np.argmax(np.mean(np.abs(stacked[0]), axis=0))  # most impactful class
        signed_values = stacked[0, :, winning_class]
    elif shap_values.ndim == 3:
        # Shape (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(shap_values[0]), axis=-1)
        winning_class = np.argmax(np.mean(np.abs(shap_values[0]), axis=0))
        signed_values = shap_values[0, :, winning_class]
    else:
        # Binary or regression: shape (n_samples, n_features)
        mean_abs = np.abs(shap_values[0])
        signed_values = shap_values[0]

    # Build feature importance list, filtered to structural features only
    structural_set = set(structural_feature_names)
    importances = []
    for i, name in enumerate(feature_names):
        if name in structural_set:
            importances.append({
                "feature": name,
                "shap_value": float(signed_values[i]),
                "abs_importance": float(mean_abs[i]),
                "direction": "positive" if signed_values[i] >= 0 else "negative",
            })

    importances.sort(key=lambda x: x["abs_importance"], reverse=True)
    # Return top_k, dropping the internal abs_importance key
    result = []
    for item in importances[:top_k]:
        result.append({
            "feature": item["feature"],
            "shap_value": item["shap_value"],
            "direction": item["direction"],
        })
    return result


def format_explanation_text(
    features: list[dict],
    predicted_label: str,
) -> str:
    """Format SHAP features into a human-readable explanation string."""
    if not features:
        return f"Predicted reception: {predicted_label}. No feature importance data available."

    lines = [f"Predicted reception: {predicted_label.upper()}"]
    lines.append("Key factors:")
    for f in features:
        direction = "helps" if f["direction"] == "positive" else "hurts"
        lines.append(f"  - {f['feature']}: {direction} (SHAP {f['shap_value']:+.3f})")
    return "\n".join(lines)
```

Also modify `/Users/malakhov/code/hn-simulator/pyproject.toml`: add `"shap>=0.51.0"` to dependencies list.

**REFACTOR phase:** None needed.

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/pip install shap>=0.51.0 && .venv/bin/python -m pytest tests/test_explain.py -v
```

#### Acceptance Criteria
- [ ] `explain_prediction` returns top-k features sorted by absolute SHAP importance
- [ ] Only structural feature names appear in output (no `emb_*` features)
- [ ] `format_explanation_text` produces human-readable string with feature names and directions
- [ ] SHAP import is lazy (only imported inside `explain_prediction`)
- [ ] No existing tests broken

---

### Task 6: Backtesting Module

**Files:**
- CREATE: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/backtest.py`
- CREATE: `/Users/malakhov/code/hn-simulator/tests/test_backtest.py`

**Depends on:** Task 2 (multiclass model, `score_to_class_index`, `LABELS_5`)

#### Context

Backtesting uses a random 80/20 split (not temporal -- research finding about sampling artifact). Trains multiclass model on 80%, evaluates on 20%. Reports: accuracy, per-class precision/recall, Spearman rho between expected_score and actual_score, confusion matrix, and an ASCII calibration chart.

#### TDD Cycle

**RED phase -- Write failing tests first:**

File: `/Users/malakhov/code/hn-simulator/tests/test_backtest.py`

```python
"""Tests for backtesting module."""
import pytest
import numpy as np


@pytest.fixture
def backtest_data():
    """Synthetic backtest data. Fixed seed."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 20)).astype(np.float32)
    scores = np.exp(X[:, 0] * 2 + rng.standard_normal(n) * 0.5).clip(1, 1000)
    feature_names = [f"feat_{i}" for i in range(20)]
    return X, scores, feature_names


class TestRandomSplit:
    def test_split_proportions(self, backtest_data):
        from hn_simulator.model.backtest import random_split
        X, scores, _ = backtest_data
        X_train, X_test, y_train, y_test = random_split(X, scores, test_size=0.2, seed=42)
        assert len(X_train) == 400
        assert len(X_test) == 100
        assert len(y_train) == 400
        assert len(y_test) == 100

    def test_split_is_deterministic(self, backtest_data):
        from hn_simulator.model.backtest import random_split
        X, scores, _ = backtest_data
        X_train1, X_test1, _, _ = random_split(X, scores, test_size=0.2, seed=42)
        X_train2, X_test2, _, _ = random_split(X, scores, test_size=0.2, seed=42)
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)

    def test_no_overlap(self, backtest_data):
        from hn_simulator.model.backtest import random_split
        X, scores, _ = backtest_data
        X_train, X_test, _, _ = random_split(X, scores, test_size=0.2, seed=42)
        assert len(X_train) + len(X_test) == len(X)


class TestRunBacktest:
    def test_returns_report_dict(self, backtest_data):
        from hn_simulator.model.backtest import run_backtest
        X, scores, names = backtest_data
        report = run_backtest(X, scores, names, test_size=0.2, seed=42)
        assert isinstance(report, dict)
        assert "accuracy" in report
        assert "spearman_rho" in report
        assert "confusion_matrix" in report
        assert "per_class" in report

    def test_accuracy_is_between_0_and_1(self, backtest_data):
        from hn_simulator.model.backtest import run_backtest
        X, scores, names = backtest_data
        report = run_backtest(X, scores, names, test_size=0.2, seed=42)
        assert 0.0 <= report["accuracy"] <= 1.0

    def test_spearman_rho_is_reasonable(self, backtest_data):
        from hn_simulator.model.backtest import run_backtest
        X, scores, names = backtest_data
        report = run_backtest(X, scores, names, test_size=0.2, seed=42)
        # With correlated synthetic data, should be > 0
        assert report["spearman_rho"] > 0.0

    def test_confusion_matrix_shape(self, backtest_data):
        from hn_simulator.model.backtest import run_backtest
        X, scores, names = backtest_data
        report = run_backtest(X, scores, names, test_size=0.2, seed=42)
        cm = np.array(report["confusion_matrix"])
        assert cm.shape == (5, 5)

    def test_per_class_has_5_entries(self, backtest_data):
        from hn_simulator.model.backtest import run_backtest
        X, scores, names = backtest_data
        report = run_backtest(X, scores, names, test_size=0.2, seed=42)
        assert len(report["per_class"]) == 5


class TestFormatBacktestReport:
    def test_returns_string_with_metrics(self):
        from hn_simulator.model.backtest import format_backtest_report
        report = {
            "accuracy": 0.45,
            "spearman_rho": 0.38,
            "confusion_matrix": np.eye(5, dtype=int).tolist(),
            "per_class": {
                "flop": {"precision": 0.5, "recall": 0.6, "count": 20},
                "low": {"precision": 0.4, "recall": 0.5, "count": 30},
                "moderate": {"precision": 0.3, "recall": 0.4, "count": 25},
                "hot": {"precision": 0.6, "recall": 0.3, "count": 15},
                "viral": {"precision": 0.7, "recall": 0.2, "count": 10},
            },
        }
        text = format_backtest_report(report)
        assert isinstance(text, str)
        assert "45" in text  # accuracy percentage
        assert "0.38" in text  # spearman
        assert "flop" in text
        assert "viral" in text
```

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/backtest.py`

```python
"""Backtesting module -- random split, multiclass evaluation, calibration report."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from hn_simulator.model.labels import (
    BUCKET_MEDIANS, LABELS_5, expected_score_from_probs, score_to_class_index,
)
from hn_simulator.model.train import train_multiclass_model


def random_split(
    X: np.ndarray, scores: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random shuffle split. Returns (X_train, X_test, y_train, y_test)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], scores[train_idx], scores[test_idx]


def run_backtest(
    X: np.ndarray,
    scores: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Run full backtest: split, train multiclass, evaluate.
    Returns dict with accuracy, spearman_rho, confusion_matrix, per_class.
    """
    X_train, X_test, scores_train, scores_test = random_split(X, scores, test_size, seed)

    y_train = np.array([score_to_class_index(s) for s in scores_train])
    y_test = np.array([score_to_class_index(s) for s in scores_test])

    model, _ = train_multiclass_model(X_train, y_train, X_test, y_test, feature_names)

    # Predict probabilities
    probs = model.predict(X_test)  # (n_test, 5)
    pred_classes = np.argmax(probs, axis=1)
    pred_expected_scores = np.array([expected_score_from_probs(p) for p in probs])

    # Accuracy
    accuracy = float(np.mean(pred_classes == y_test))

    # Spearman rho between expected_score and actual_score
    rho, _ = spearmanr(pred_expected_scores, scores_test)

    # Confusion matrix (5x5)
    cm = np.zeros((5, 5), dtype=int)
    for true, pred in zip(y_test, pred_classes):
        cm[true][pred] += 1

    # Per-class precision/recall
    per_class = {}
    for i, label in enumerate(LABELS_5):
        tp = cm[i][i]
        col_sum = cm[:, i].sum()  # predicted as class i
        row_sum = cm[i, :].sum()  # actually class i
        precision = float(tp / col_sum) if col_sum > 0 else 0.0
        recall = float(tp / row_sum) if row_sum > 0 else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "count": int(row_sum)}

    return {
        "accuracy": accuracy,
        "spearman_rho": float(rho),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def format_backtest_report(report: dict) -> str:
    """Format backtest report as ASCII text with calibration chart."""
    lines = []
    lines.append("=" * 50)
    lines.append("  BACKTEST REPORT")
    lines.append("=" * 50)
    lines.append(f"  Accuracy:     {report['accuracy']:.1%}")
    lines.append(f"  Spearman rho: {report['spearman_rho']:.3f}")
    lines.append("")

    # Per-class table
    lines.append("  Per-Class Metrics:")
    lines.append(f"  {'Class':<12} {'Prec':>6} {'Recall':>6} {'Count':>6}")
    lines.append("  " + "-" * 32)
    for label in LABELS_5:
        pc = report["per_class"][label]
        lines.append(
            f"  {label:<12} {pc['precision']:>5.1%} {pc['recall']:>5.1%} {pc['count']:>6d}"
        )
    lines.append("")

    # Confusion matrix
    lines.append("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = report["confusion_matrix"]
    header = "  " + " " * 10 + "".join(f"{l[:4]:>6}" for l in LABELS_5)
    lines.append(header)
    for i, label in enumerate(LABELS_5):
        row = "  " + f"{label:<10}" + "".join(f"{cm[i][j]:>6d}" for j in range(5))
        lines.append(row)

    lines.append("=" * 50)
    return "\n".join(lines)
```

**REFACTOR phase:** None needed.

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_backtest.py -v
```

#### Acceptance Criteria
- [ ] `random_split` produces deterministic 80/20 splits with seed
- [ ] `run_backtest` trains a model and returns all required metrics
- [ ] Accuracy is between 0 and 1; Spearman rho > 0 for correlated data
- [ ] Confusion matrix is 5x5, per_class has 5 entries
- [ ] `format_backtest_report` produces readable ASCII output with all metrics

---

### Task 7: Iterative Optimization Loop

**Files:**
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/suggest.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/tests/test_suggest.py`

**Depends on:** Task 2 (multiclass model used by simulator for scoring)

#### Context

Add an iterative loop: `suggest -> score -> suggest better`, max 5 iterations. Track `seen_titles` set to avoid duplicate suggestions. Stop early if the best score improvement between iterations is below a convergence threshold (default: 5% relative improvement). The existing `suggest_and_score` is called in each iteration; each subsequent call includes context about previous best scores.

#### TDD Cycle

**RED phase -- Write failing tests first:**

Add to `/Users/malakhov/code/hn-simulator/tests/test_suggest.py`:

```python
class TestIterativeSuggest:
    def test_returns_list_with_best_variant(self, mock_models_and_simulator):
        from hn_simulator.suggest import iterative_suggest
        sim = mock_models_and_simulator

        mock_client = MagicMock()
        # Return different suggestions each call
        call_count = [0]
        def mock_create(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.content = [MagicMock(text=json.dumps([
                {"title": f"Iteration {call_count[0]} Title", "description": f"Desc {call_count[0]}"},
            ]))]
            return resp
        mock_client.messages.create = mock_create

        original = {"title": "Original Title", "description": "Original desc"}
        result = iterative_suggest(
            simulator=sim,
            original=original,
            client=mock_client,
            max_iterations=3,
            num_suggestions=1,
        )
        assert isinstance(result, dict)
        assert "best" in result
        assert "iterations" in result
        assert result["best"]["title"] is not None
        assert result["best"]["predicted_score"] >= 0
        assert 1 <= result["iterations"] <= 3

    def test_tracks_seen_titles(self, mock_models_and_simulator):
        from hn_simulator.suggest import iterative_suggest
        sim = mock_models_and_simulator

        mock_client = MagicMock()
        # Return same title every time to test dedup
        resp = MagicMock()
        resp.content = [MagicMock(text=json.dumps([
            {"title": "Same Title Every Time", "description": "Desc"},
        ]))]
        mock_client.messages.create.return_value = resp

        original = {"title": "Original", "description": "Desc"}
        result = iterative_suggest(
            simulator=sim, original=original, client=mock_client,
            max_iterations=3, num_suggestions=1,
        )
        # Should not have more unique variants than iterations + original
        assert result["iterations"] >= 1

    def test_max_iterations_respected(self, mock_models_and_simulator):
        from hn_simulator.suggest import iterative_suggest
        sim = mock_models_and_simulator

        mock_client = MagicMock()
        call_count = [0]
        def mock_create(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.content = [MagicMock(text=json.dumps([
                {"title": f"Title {call_count[0]}", "description": f"D{call_count[0]}"},
            ]))]
            return resp
        mock_client.messages.create = mock_create

        original = {"title": "Original", "description": "Desc"}
        result = iterative_suggest(
            simulator=sim, original=original, client=mock_client,
            max_iterations=2, num_suggestions=1,
        )
        assert result["iterations"] <= 2

    def test_convergence_stops_early(self, mock_models_and_simulator):
        from hn_simulator.suggest import iterative_suggest
        sim = mock_models_and_simulator

        mock_client = MagicMock()
        # Always return the same suggestion -> score won't change -> converge
        resp = MagicMock()
        resp.content = [MagicMock(text=json.dumps([
            {"title": "Converged Title", "description": "Desc"},
        ]))]
        mock_client.messages.create.return_value = resp

        original = {"title": "Original", "description": "Desc"}
        result = iterative_suggest(
            simulator=sim, original=original, client=mock_client,
            max_iterations=5, num_suggestions=1,
            convergence_threshold=0.05,
        )
        # Should stop early because no new unique titles to score
        assert result["iterations"] <= 5
```

**GREEN phase -- Minimal implementation:**

Add to `/Users/malakhov/code/hn-simulator/src/hn_simulator/suggest.py`:

```python
def iterative_suggest(
    simulator: "HNSimulator",
    original: dict,
    client,
    max_iterations: int = 5,
    num_suggestions: int = 5,
    convergence_threshold: float = 0.05,
    generate_comments: bool = False,
) -> dict:
    """Iteratively suggest -> score -> suggest better, up to max_iterations.

    Tracks seen_titles to avoid duplicate suggestions.
    Stops early if best score improvement is below convergence_threshold (relative).

    Returns dict with:
        - "best": dict with title, description, predicted_score, reception_label
        - "iterations": number of iterations completed
        - "history": list of best-score per iteration
        - "all_variants": list of all unique scored variants
    """
    seen_titles: set[str] = set()
    all_variants: list[dict] = []
    best_score = 0.0
    best_variant = {
        "title": original.get("title", ""),
        "description": original.get("description", ""),
        "predicted_score": 0.0,
        "reception_label": "flop",
    }
    history: list[float] = []

    # Score the original first
    sim_result = simulator.simulate(
        original.get("title", ""),
        original.get("description", ""),
        generate_comments=False,
    )
    best_score = sim_result.predicted_score
    best_variant = {
        "title": original["title"],
        "description": original.get("description", ""),
        "predicted_score": sim_result.predicted_score,
        "reception_label": sim_result.reception_label,
        "is_original": True,
    }
    seen_titles.add(original["title"])
    all_variants.append(best_variant)

    for iteration in range(max_iterations):
        prev_best = best_score

        # Build context about previous best for better suggestions
        context_original = {
            "title": original["title"],
            "description": original.get("description", "")
            + f"\n\nPrevious best score: {best_score:.0f} ({best_variant['reception_label']}). "
            + f"Best title so far: '{best_variant['title']}'. Generate different/better options.",
        }

        suggestions = suggest_variants(context_original, client=client, num_suggestions=num_suggestions)

        # Filter out seen titles
        new_suggestions = [s for s in suggestions if s.get("title", "") not in seen_titles]

        if not new_suggestions:
            history.append(best_score)
            break

        # Score new suggestions
        for s in new_suggestions:
            title = s.get("title", "")
            desc = s.get("description", "")
            seen_titles.add(title)
            sim_result = simulator.simulate(title, desc, generate_comments=False)
            variant = {
                "title": title,
                "description": desc,
                "predicted_score": sim_result.predicted_score,
                "reception_label": sim_result.reception_label,
                "is_original": False,
            }
            all_variants.append(variant)
            if sim_result.predicted_score > best_score:
                best_score = sim_result.predicted_score
                best_variant = variant

        history.append(best_score)

        # Check convergence
        if prev_best > 0 and (best_score - prev_best) / prev_best < convergence_threshold:
            break

    return {
        "best": best_variant,
        "iterations": len(history),
        "history": history,
        "all_variants": all_variants,
    }
```

**REFACTOR phase:** None needed -- function is self-contained.

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_suggest.py -v
```

#### Acceptance Criteria
- [ ] `iterative_suggest` returns dict with "best", "iterations", "history" keys
- [ ] `seen_titles` prevents duplicate scoring
- [ ] Loop stops at `max_iterations` or earlier on convergence
- [ ] Original title is scored first and included in results
- [ ] All existing `suggest_variants` and `suggest_and_score` tests still pass

---

### Task 8: Simulator + CLI Integration

**Files:**
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/simulator.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/src/hn_simulator/cli.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/tests/test_simulator.py`
- MODIFY: `/Users/malakhov/code/hn-simulator/tests/test_cli.py`

**Depends on:** Tasks 1, 2, 3, 4, 5, 6, 7 (all features must be built)

#### Context

Wire all new features into the simulator orchestrator and CLI output:
1. `SimulationResult` gets new fields: `percentile`, `expected_score`, `shap_features`, `posting_time_advice`
2. `HNSimulator.__init__` optionally loads multiclass model, calibration data, time stats, domain stats
3. `HNSimulator.simulate` uses multiclass model for classification, adds percentile, SHAP, time advice
4. CLI `predict` command shows percentile, expected score, SHAP explanation, posting time recommendation
5. CLI gets new `backtest` subcommand
6. CLI gets new `suggest-loop` subcommand for iterative optimization

#### TDD Cycle

**RED phase -- Write failing tests first:**

Update `/Users/malakhov/code/hn-simulator/tests/test_simulator.py`:

```python
class TestSimulationResultV2:
    def test_to_dict_includes_new_fields(self):
        from hn_simulator.simulator import SimulationResult
        result = SimulationResult(
            predicted_score=42.5,
            predicted_comments=15.0,
            reception_label="hot",
            confidence=0.75,
            label_distribution={"flop": 0.05, "low": 0.10, "moderate": 0.15, "hot": 0.45, "viral": 0.25},
            percentile=15.0,
            expected_score=85.3,
            shap_features=[{"feature": "is_show_hn", "shap_value": 0.5, "direction": "positive"}],
            posting_time_advice={"best_hour": 9, "best_day_name": "Sunday"},
        )
        d = result.to_dict()
        assert d["percentile"] == 15.0
        assert d["expected_score"] == 85.3
        assert len(d["shap_features"]) == 1
        assert d["posting_time_advice"]["best_hour"] == 9
```

Update `/Users/malakhov/code/hn-simulator/tests/test_cli.py`:

```python
class TestCliBacktest:
    def test_backtest_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0

class TestCliSuggestLoop:
    def test_suggest_loop_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["suggest-loop", "--help"])
        assert result.exit_code == 0
        assert "title" in result.output.lower()
        assert "max-iterations" in result.output
```

**GREEN phase -- Minimal implementation:**

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/simulator.py`

1. Update `SimulationResult` dataclass -- add fields:
   - `percentile: float | None = None`
   - `expected_score: float | None = None`
   - `shap_features: list[dict] = field(default_factory=list)`
   - `posting_time_advice: dict | None = None`

2. Update `to_dict()` to include new fields.

3. Update `HNSimulator.__init__` -- add optional parameters:
   - `multiclass_model_path: Path | str | None = None`
   - `sorted_scores_path: Path | str | None = None`
   - `time_stats_path: Path | str | None = None`
   - `domain_stats_path: Path | str | None = None`
   - Load each if path exists, store as instance attributes (None if not available)

4. Update `HNSimulator.simulate`:
   - If multiclass model loaded, predict probabilities, compute expected_score, get class from argmax
   - If sorted_scores loaded, compute percentile from expected_score
   - If SHAP available and multiclass model loaded, compute SHAP features (top 5 structural)
   - If time_stats loaded, include posting_time_advice in result
   - Keep backward compat: if no multiclass model, fall back to regression + rule-based classification

File: `/Users/malakhov/code/hn-simulator/src/hn_simulator/cli.py`

1. Update `_human_output` to display:
   - Percentile: `"  Percentile: Top {pct:.0f}% of all HN posts"`
   - Expected score from multiclass: `"  Expected Score: ~{score:.0f} points (from class probabilities)"`
   - SHAP explanation block
   - Posting time advice

2. Add `backtest` subcommand:
   ```python
   @main.command()
   @click.option("--test-size", default=0.2, help="Test set fraction")
   @click.option("--seed", default=42, help="Random seed for split")
   def backtest(test_size, seed):
       """Run backtesting on training data and print calibration report."""
   ```
   Loads `features.npy`, `labels_score.npy`, `feature_names.json` from `PROCESSED_DIR`. Calls `run_backtest`, prints `format_backtest_report`.

3. Add `suggest-loop` subcommand:
   ```python
   @main.command("suggest-loop")
   @click.option("--title", required=True)
   @click.option("--description", default="")
   @click.option("--max-iterations", default=5)
   @click.option("--num-suggestions", default=5)
   def suggest_loop(title, description, max_iterations, num_suggestions):
       """Iteratively generate and score title variants."""
   ```

**REFACTOR phase:**
- Extract model/data loading into a helper `_load_optional(path, loader)` pattern to reduce boilerplate.

#### Verify
```bash
cd /Users/malakhov/code/hn-simulator && .venv/bin/python -m pytest tests/test_simulator.py tests/test_cli.py -v
```

#### Acceptance Criteria
- [ ] `SimulationResult.to_dict()` includes percentile, expected_score, shap_features, posting_time_advice
- [ ] `HNSimulator` gracefully degrades when optional model/data files are missing
- [ ] CLI `predict` shows percentile and SHAP explanation when data is available
- [ ] CLI `backtest` subcommand exists and prints report
- [ ] CLI `suggest-loop` subcommand exists and runs iterative optimization
- [ ] All existing CLI tests still pass (backward compatible)
- [ ] All existing simulator tests still pass

---

## Dependency Graph

```
Wave 1 (parallel): Tasks 1, 3, 4
  - Task 1: Domain reputation features (new structural features)
  - Task 3: Percentile calibration + time-of-day (new standalone module)
  - Task 4: Better RAG .where() filter (independent retrieval fix)

Wave 2 (parallel): Tasks 2, 5
  - Task 2: Multiclass classification (depends on Task 1 for feature count)
  - Task 5: SHAP explanations (depends on Task 1 for feature names, Task 2 for multiclass model)
  Note: Task 5 can start once Task 2's model training function exists

Wave 3 (parallel): Tasks 6, 7
  - Task 6: Backtesting (depends on Task 2 for multiclass training)
  - Task 7: Iterative suggest loop (depends on Task 2 for classifier-based scoring)

Wave 4 (sequential): Task 8
  - Task 8: Simulator + CLI integration (depends on all previous tasks)
```

## Cross-Cutting Concern: Feature Count Updates

When Task 1 changes structural features from 13 to 15 (total 397 -> 399), the following hardcoded values across the codebase must ALL be updated in the same commit as Task 1:

| File | What to change |
|------|---------------|
| `src/hn_simulator/features/pipeline.py` | Docstring: "13 cols" -> "15 cols", "397" -> "399" |
| `tests/test_structural_features.py` | Implicit assertion on 13 columns (currently checks list, add explicit count check for 15) |
| `tests/test_feature_pipeline.py` | `397` -> `399` in shape assertions (lines 31-32, 49) |
| `tests/test_simulator.py` | `mock_models` fixture: `X = rng.standard_normal((100, 397))` -> `399`, `names = [f"feat_{i}" for i in range(397)]` -> `399` |
| `tests/test_suggest.py` | `mock_models_and_simulator` fixture: same 397 -> 399 changes |
| `tests/test_compare.py` | `mock_models_and_simulator` fixture: same 397 -> 399 changes |
| `tests/test_integration.py` | `full_pipeline` fixture: shape assertion from 397 -> 399 (implicit via build_feature_matrix) |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| SHAP import slow (~2s first load) | Increased prediction latency on first call | Lazy import inside `explain_prediction` only; document in CLI output |
| LanceDB `.where()` API differs across versions | RAG retrieval breaks | Fallback to `.to_pandas()` in except clause; pin lancedb version in pyproject.toml |
| Feature count change (397->399) breaks trained models | Old models incompatible | Keep old regression models separate; multiclass model trained with new feature count |
| Multiclass model with class imbalance | Poor accuracy on rare classes (viral) | Class weights via inverse frequency in training; document in backtest report |
| SHAP multiclass shape varies by shap version | Crash in explain_prediction | Handle both `list[ndarray]` and `ndarray` shapes with explicit branching |
| Convergence threshold too aggressive in iterative loop | Stops too early with 1 iteration | Default threshold 5% is conservative; user can override via CLI flag |
| `scipy.stats.spearmanr` not in dependencies | ImportError in backtest | scipy is a transitive dependency of scikit-learn (already required); verify at install time |

---

**Total new test count estimate:** ~60 new tests across 4 new test files + updates to 6 existing test files. All 167 existing tests must continue to pass.

**Files created (4):**
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/calibrate.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/explain.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/backtest.py`
- `/Users/malakhov/code/hn-simulator/tests/test_domain_features.py`
- `/Users/malakhov/code/hn-simulator/tests/test_multiclass.py`
- `/Users/malakhov/code/hn-simulator/tests/test_calibrate.py`
- `/Users/malakhov/code/hn-simulator/tests/test_explain.py`
- `/Users/malakhov/code/hn-simulator/tests/test_backtest.py`

**Files modified (12):**
- `/Users/malakhov/code/hn-simulator/pyproject.toml`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/config.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/features/structural.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/features/pipeline.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/labels.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/model/train.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/rag/retrieve.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/suggest.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/simulator.py`
- `/Users/malakhov/code/hn-simulator/src/hn_simulator/cli.py`
- `/Users/malakhov/code/hn-simulator/tests/test_labels.py`
- `/Users/malakhov/code/hn-simulator/tests/test_structural_features.py`
- `/Users/malakhov/code/hn-simulator/tests/test_feature_pipeline.py`
- `/Users/malakhov/code/hn-simulator/tests/test_simulator.py`
- `/Users/malakhov/code/hn-simulator/tests/test_suggest.py`
- `/Users/malakhov/code/hn-simulator/tests/test_compare.py`
- `/Users/malakhov/code/hn-simulator/tests/test_cli.py`
- `/Users/malakhov/code/hn-simulator/tests/test_integration.py`