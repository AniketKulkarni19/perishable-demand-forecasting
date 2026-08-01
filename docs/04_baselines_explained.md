# Phase 4 — Validation Strategy, Metrics, and Baselines

**Notebook:** `notebooks/04_baselines.ipynb`
**Input:** `data/processed/protein_features.parquet` (7.46 M rows, 36 columns)
**Purpose:** Establish the honest benchmark that every later model must beat.

---

## 1. The time-based split — the most important decision in the project

```python
TEST_START  = "2017-07-31"
VALID_START = "2017-07-15"

train = df[df["date"] <  VALID_START].copy()
valid = df[(df["date"] >= VALID_START) & (df["date"] < TEST_START)].copy()
test  = df[df["date"] >= TEST_START].copy()
```

| Set | Rows | Date range | Share |
|---|---|---|---|
| Train | 7,205,999 | 2015-01-29 → 2017-07-14 | 96.6% |
| Valid | 129,573 | 2017-07-15 → 2017-07-30 | 1.7% |
| Test | 124,039 | 2017-07-31 → 2017-08-15 | 1.7% |

### Why `train_test_split()` would be catastrophically wrong

A random split — the default in essentially every classification tutorial, and what most coursework uses — scatters future dates into the training set. The model then learns from days that occur **after** the days it is predicting.

The result: **excellent validation scores and total production failure.** Anyone who knows forecasting checks for this specifically when reviewing a portfolio project.

**The rule: train on the past, validate on the future, never let them overlap.**

### Why the split looks so lopsided
96.6% train share is correct here, and not a mistake:
- Maximum history is valuable — the model needs multiple seasonal cycles
- The evaluation window is defined by the **business horizon** (a 16-day forecast), not by an arbitrary percentage
- 16 days mirrors the original Kaggle competition's horizon, making results comparable to a real benchmark

### Why three sets and not two

| Set | How often it's used |
|---|---|
| **train** | Model fitting |
| **valid** | Repeatedly — comparing models, tuning hyperparameters |
| **test** | **Exactly once**, at the very end |

Every time you look at the test set and then adjust something, information leaks into your decisions and the number stops being an honest estimate of future performance. The discipline of touching test once is what makes the final number meaningful.

---

## 2. Metrics — and why MAPE is unusable here

### The metrics we use

```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
```

| Metric | What it measures | Why it's here |
|---|---|---|
| **MAE** | Average absolute error in units | Directly interpretable: "off by 3.97 units per store-item-day" |
| **RMSE** | Squares errors before averaging | Large misses hurt disproportionately — catches catastrophic days that MAE smooths over |
| **WAPE** | Total error ÷ total actual | The percentage metric that survives zeros. **The headline business number** |
| **RMSLE** | Error on a log scale | The original competition metric. Penalizes under-forecasting more than over-forecasting; handles heavy right skew |

### Why MAPE is excluded

MAPE (Mean Absolute Percentage Error) is the metric most people reach for by default. **It cannot be used on this dataset.**

MAPE divides by the actual value for each row. With 23% of rows being zero-demand days:
- Division by zero → infinity
- Even near-zero actuals (0.5 units) produce absurd percentages that dominate the average

**WAPE solves this** by aggregating first and dividing once: `sum(|error|) / sum(actual)`. The denominator is the total, which is never zero.

Recognising this trap is itself a signal of competence in intermittent demand forecasting — it's a very common mistake.

### Why `log1p` in RMSLE
`log1p(x)` = `log(1 + x)`, which is defined at zero. Plain `log(0)` is undefined. The `+1` makes the metric safe for zero-demand rows.

---

## 3. Baseline results

```
                 model    MAE    RMSE    WAPE   RMSLE
         Moving avg 7d  3.9725  13.6616  49.95  0.7042
        Moving avg 28d  3.9751  13.5639  49.99  0.7073
Seasonal naive (lag_7)  4.2901  10.2457  53.95  0.8342
         Naive (lag_1)  5.0158  21.1070  63.07  0.8568
           Always zero  7.9522  21.1992 100.00  1.8665
```

### The baselines are free
The features engineered in Phase 3 **are** the baselines — `lag_1` is literally the naive forecast, `roll_mean_7` is literally the moving average. No extra computation was needed.

### What each baseline represents

| Baseline | Prediction |
|---|---|
| **Naive** | Tomorrow = today |
| **Seasonal naive** | Next Saturday = last Saturday |
| **Moving average** | Tomorrow = recent average |
| **Always zero** | Deliberately stupid sanity floor |

### "Always zero" scores exactly 100% WAPE

This is **mathematically necessary**: `sum(|y − 0|) / sum(y) = 1` when all predictions are zero.

Hitting exactly 100.00 confirms the WAPE implementation is correct. Including a deliberately worthless model is a cheap and effective test of your evaluation code.

### The interesting finding: the metrics disagree

**Moving average wins on MAE and WAPE. Seasonal naive wins decisively on RMSE** (10.25 vs 13.66).

The reason:
- The **moving average smooths everything**, so it badly misses weekly spikes (Saturday demand). RMSE squares those large misses, so it punishes them heavily.
- **Seasonal naive compares Saturday to last Saturday**, so it *captures* the weekly spike. It's noisier on ordinary days (worse MAE) but avoids catastrophic errors (better RMSE).

**This tension is exactly what a gradient boosting model should resolve** — it can learn both the smooth demand level and the weekly pattern at the same time, rather than trading one for the other.

### Context for the ~50% WAPE

A WAPE near 50% looks alarming in isolation. It is **normal for daily SKU-store-level intermittent demand**:
- Predicting a single item at a single store on a single day is inherently noisy
- 23% of the target values are zero
- Aggregating to weekly totals or category level drops WAPE sharply

This should be stated explicitly in the README so the number isn't misread as poor modelling.

---

## 4. The benchmark to beat

```
WAPE  49.95   (Moving avg 7d)
MAE    3.97   (Moving avg 7d)
RMSE  10.25   (Seasonal naive)
RMSLE  0.704  (Moving avg 7d)
```

Every subsequent model is measured against these numbers.

### Why baselines are non-negotiable

Without a baseline, a model's metrics are meaningless. "WAPE of 42%" says nothing on its own. "WAPE of 42% against a 49.95% moving-average baseline — a 16% improvement" is a result.

Skipping baselines is one of the clearest signals of an inexperienced portfolio project. A sophisticated model that fails to beat a moving average is a **negative result worth reporting**, not something to hide.

---

## Key takeaways

- *"I used a strictly time-based split rather than a random one. A random split leaks future information into training, which produces excellent validation scores and complete production failure — it's the most common fatal error in time-series work."*
- *"I excluded MAPE because 23% of the target values are zero, which makes it undefined. WAPE aggregates before dividing, so it survives intermittent demand — that's the headline business metric."*
- *"The best baseline is a 7-day moving average at 49.95% WAPE. Every model I built afterwards is reported against that number, so the improvement is measurable rather than asserted."*
- *"Interestingly the metrics disagreed on the best baseline: the moving average won on MAE and WAPE, but seasonal naive won on RMSE because it captures weekly spikes the moving average smooths away. That tension is what a gradient boosting model should be able to resolve."*
- *"I included a deliberately worthless 'always predict zero' baseline as a sanity check. It scored exactly 100% WAPE, which mathematically confirms the metric implementation is correct."*

---

## Pipeline state

```
data/processed/protein_features.parquet   7.46 M rows, 36 columns
  ↓ time-based split
train  7,205,999   2015-01-29 → 2017-07-14
valid    129,573   2017-07-15 → 2017-07-30
test     124,039   2017-07-31 → 2017-08-15   ← touched once, at the end
```

**Next:** holiday features (measurable lift against these baselines), then LightGBM.
