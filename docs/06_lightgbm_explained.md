# Phase 6 — LightGBM: Training, Results, and What Actually Mattered

**Notebook:** `notebooks/06_lightgbm.ipynb`
**Input:** `protein_features_hol.parquet` (7.46 M rows, 45 columns)

---

## Headline result

| Model | MAE | RMSE | WAPE | RMSLE |
|---|---|---|---|---|
| **LightGBM (with holidays)** | **3.2528** | **8.2649** | **40.90** | **0.5866** |
| LightGBM (no holidays) | 3.2617 | 8.7001 | 41.02 | 0.5867 |
| Moving avg 7d (baseline) | 3.9725 | 13.6616 | 49.95 | 0.7042 |
| Seasonal naive (baseline) | 4.2901 | 10.2457 | 53.95 | 0.8342 |

**Improvement over the best baseline: WAPE 49.95 → 40.90, a 17.9% reduction in forecast error.**

The model also beat the moving average on MAE/WAPE *and* seasonal naive on RMSE — resolving the tension the baselines exposed, where no single naive method won on both.

---

## Setup notes

### macOS install issue
LightGBM's compiled library requires OpenMP, which macOS does not ship:
```
OSError: Library not loaded: @rpath/libomp.dylib
```
Fix: `brew install libomp`, then **restart the Jupyter kernel** — the library is loaded at import time, so a running kernel will not pick it up.

### Why `store_nbr` and `item_nbr` stay as features

Kept as **categorical** features rather than dropped as identifiers. This lets the model learn store-specific and item-specific demand levels directly.

LightGBM handles high-cardinality categoricals natively — 263 items and 54 stores are fine. One-hot encoding would have added 300+ sparse columns for the same information.

### Why `log1p` on the target

```python
y_train_log = np.log1p(y_train)
```

Sales are heavily right-skewed — mostly small values with occasional large spikes. Training on raw values makes the model chase those spikes at the expense of typical days. The log transform compresses the scale and aligns with RMSLE, the competition metric.

Predictions are inverted with `np.expm1()` and floored at zero before scoring:
```python
pred = np.clip(np.expm1(model.predict(X)), 0, None)
```
The floor matters — the model can predict slightly below zero in log space, which is meaningless for demand.

---

## Hyperparameters and reasoning

```python
params = {
    "objective": "regression",     # L2 loss, appropriate on log-transformed target
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 63,              # model complexity
    "min_data_in_leaf": 100,       # prevents splitting on noise
    "feature_fraction": 0.8,       # each tree sees 80% of features
    "bagging_fraction": 0.8,       # each tree sees 80% of rows
    "bagging_freq": 1,
    "num_threads": 4,
    "seed": 42,                    # reproducibility
}
```

`feature_fraction` and `bagging_fraction` introduce randomness that reduces overfitting — each tree sees a different slice, so they make different mistakes and average out better.

### Early stopping
```python
callbacks=[lgb.early_stopping(30)]
```
Training halts if validation RMSE has not improved for 30 rounds. **This is what removes the need to guess the right number of trees** — the validation set decides.

### ⚠️ The model is undertrained
```
Did not meet early stopping. Best iteration is: [400]
```
Validation RMSE was still improving at round 400 (0.5875 → 0.5868). More boosting rounds would likely help, though gains are clearly flattening.

**This is the most obvious next tuning step** and should be stated as a known limitation rather than left unmentioned.

---

## The controlled experiment: do holiday features help?

Identical parameters, identical seed, **only the feature set differs**. That isolation is what makes the comparison meaningful.

| Metric | No holidays | With holidays | Change |
|---|---|---|---|
| WAPE | 41.02 | 40.90 | −0.3% |
| MAE | 3.2617 | 3.2528 | −0.3% |
| **RMSE** | **8.7001** | **8.2649** | **−5.0%** |
| RMSLE | 0.5867 | 0.5866 | ~0 |

### The honest reading

**On typical-day accuracy, holiday features contributed essentially nothing** — a 0.3% WAPE change is within noise.

**On RMSE they produced a real 5% improvement.** Because RMSE squares errors, this means holidays help avoid *catastrophic misses* rather than improving the average day — consistent with them mattering on a handful of days per year rather than across the board.

**Why keep them:** for perishable protein, large errors are where spoilage cost concentrates. A feature set that reduces the worst misses by 5% has business value even when average accuracy is unchanged. That is the defensible argument — not a claim of general improvement.

---

## Feature importance (gain)

"Gain" measures how much each feature reduced loss across all splits — the honest measure of contribution.

| Rank | Feature | Gain % |
|---|---|---|
| 1 | roll_mean_14 | **50.93** |
| 2 | roll_mean_28 | **22.11** |
| 3 | roll_mean_7 | **7.26** |
| 4 | lag_7 | 3.52 |
| 5 | item_nbr | 2.06 |
| 6 | lag_14 | 1.77 |
| 7 | lag_1 | 1.68 |
| 8 | lag_28 | 1.64 |
| 9 | onpromotion | 1.59 |
| 10 | dayofweek | 1.54 |
| 11 | days_since_sale | 1.20 |
| 12 | store_nbr | 0.78 |
| 13 | zero_rate_28 | 0.76 |

### Rolling means dominate — 80.3% of total gain

The three rolling means alone account for over four-fifths of the model's predictive power. **Demand is mostly "what it recently was."**

This is a genuinely useful finding: it says the sophisticated feature engineering matters far less than getting a good recent-average signal. Worth stating plainly rather than implying every feature contributed.

### Two predictions that were wrong

**`days_since_sale` earned its keep** (1.20%, ahead of `store_nbr`). It was flagged in Phase 3 as possibly useless because it stays near zero for daily-selling items. It turned out to matter — presumably on the intermittent SEAFOOD and slow-moving SKUs it was designed for.

**Holiday proximity features barely registered.** They were predicted to be top features. They were not.

### Holiday features, ranked

| Feature | Gain % |
|---|---|
| days_from_holiday | 0.10 |
| days_to_holiday | 0.08 |
| hol_national | 0.06 |
| is_holiday | 0.04 |
| is_event | 0.01 |
| is_earthquake | 0.00 |
| hol_local | 0.00 |
| hol_regional | 0.00 |
| **is_workday_makeup** | **0.00** — never used in a single split |

All nine holiday features combined: **0.29% of total gain.**

`is_workday_makeup` scoring exactly 0.0 means LightGBM never once found it worth splitting on. It is a genuinely dead feature.

---

## Key takeaways

- *"LightGBM reduced WAPE from 49.95% to 40.90% against the best naive baseline — a 17.9% improvement in forecast error. Reporting against a baseline is what makes that number meaningful; on its own, '41% WAPE' says nothing."*
- *"I ran a controlled comparison with and without holiday features — same parameters, same seed, only the feature set changed. They improved typical-day accuracy by 0.3%, which is noise, but reduced RMSE by 5%. Because RMSE squares errors, that means they help avoid catastrophic misses rather than the average day, which for perishables is where spoilage cost concentrates."*
- *"Three rolling means account for 80% of total model gain. The sophisticated features mattered far less than a good recent-average signal — that's worth knowing before over-investing in feature engineering."*
- *"One feature, is_workday_makeup, was never used in a single split. I built it, tested it, and it contributed nothing — that's a result worth reporting rather than quietly leaving in."*
- *"The model didn't hit early stopping at 400 rounds, so it's undertrained. More boosting rounds is the clearest remaining improvement, and I'd rather state that as a known limitation than present the current number as fully tuned."*

---

## Pipeline state and next steps

```
protein_features_hol.parquet   7.46 M rows, 45 columns
  ↓ time-based split
train 7,205,999 | valid 129,573 | test 124,039  ← test still untouched
  ↓ LightGBM, log1p target
Validation WAPE 40.90  (baseline 49.95)
```

**Remaining work:**
1. More boosting rounds / hyperparameter tuning
2. SHAP explainability
3. Business impact translation — WAPE into spoilage cost in dollars
4. Final evaluation on the **test set, touched exactly once**
5. FastAPI + Docker deployment
6. README
