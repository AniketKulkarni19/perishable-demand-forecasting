# Phase 2 — Cleaning and Grid Reconstruction: What We Did and Why

**Notebook:** `notebooks/02_cleaning.ipynb`
**Input:** `data/processed/protein_sales.parquet` (9.32 M rows)
**Output:** `data/processed/protein_grid.parquet` (7.73 M rows, continuous daily series)

---

## Why a separate notebook

`01_eda.ipynb` contains the slow extract cell that reads the 4.7 GB source file. Re-running it every session to get back to a working DataFrame wastes minutes.

From Phase 2 onward, **the Parquet file is the entry point**:
```python
df = pd.read_parquet('../data/processed/protein_sales.parquet')
```
Loads in seconds. The 4.7 GB CSV is never touched again.

**General practice:** scope notebooks to a pipeline stage — EDA → cleaning → features → modeling. Each reads the previous stage's saved output. This also reads well to anyone browsing the repo.

---

## 1. Date cut: 2015-01-01 onward

### The decision
Dropped 2013–2014, keeping 5.93 M of 9.32 M rows (63.7%) and 956 days (~2.6 years).

### Why discarding 36% of the data is correct here

**Promotion is likely a top-3 predictive feature.** Promotions drive large demand spikes in grocery retail. With 21% of that column null, the options were:
- Drop the column → lose the strongest signal
- Impute "no promo" → actively teach the model that promotions don't matter, on days that *did* have promotions

Corrupting the best feature to retain older rows is a bad trade.

**Two full seasonal cycles is the real requirement.** To learn annual seasonality a model needs at least two complete cycles. 2.6 years clears that bar. Below two cycles, seasonality and trend can't be separated; above it, returns diminish quickly.

**Recency beats volume in time series.** This is the key difference from classification problems. In churn or fraud, more rows is almost always better. In forecasting, 2013 demand reflects different store counts, product mixes, and pricing regimes — stale data can actively mislead.

**README framing:**
> *"Restricted to the period with complete promotional data. Promotion is a primary demand driver, and imputing it would corrupt the signal."*

### The `.copy()` detail
```python
df = df[df["date"] >= "2015-01-01"].copy()
```
Without `.copy()`, pandas returns a *view* into the original frame, and later edits raise `SettingWithCopyWarning` (and may not apply as expected). **Filter, then copy** is the safe standard pattern.

---

## 2. Negative sales: clipped to zero

298 negative values remained after the date cut — nearly all fractional, on weight-sold proteins. These are returns/adjustments, not demand.

```python
df["unit_sales"] = df["unit_sales"].clip(lower=0)
```

| Option | Trade-off |
|---|---|
| **Clip to 0** ✓ | Forecasting *demand*, which can't be negative. Keeps the row, removes the impossible value |
| Drop rows | Loses those dates entirely, creating artificial gaps in the series |
| Leave as-is | Model learns to predict negative sales — nonsensical output |

`clip(lower=0)` ≡ `CASE WHEN unit_sales < 0 THEN 0 ELSE unit_sales END`

---

## 3. Dtype optimization: 745 MB → 187 MB (74.9% reduction)

### The principle
Pandas defaults to `int64` and `float64` — **8 bytes per value regardless of what's stored**. But `store_nbr` only reaches 54, which fits in `int8` (1 byte). Same data, one-eighth the memory.

| Type | Bytes | Range |
|---|---|---|
| int8 | 1 | −128 to 127 |
| int16 | 2 | ±32,767 |
| int32 | 4 | ±2.1 billion |
| float32 | 4 | ~7 significant digits |

### What we applied

| Column | Max value | Chosen type |
|---|---|---|
| `store_nbr` | 54 | int8 |
| `cluster` | 17 | int8 |
| `class` | 2,986 | int16 |
| `item_nbr` | 2,081,175 | int32 |
| `unit_sales` | — | float32 |
| `family`, `city`, `state`, `store_type` | — | category |

**Categorical dtype** stores integer codes plus one lookup table instead of repeating `'PREPARED FOODS'` six million times. Structurally identical to a dimension table in a star schema.

**`float32` is safe** — values like `29.904` need nowhere near float64's 15-digit precision.

### Dropping `perishable`
Every row had value 1 (we filtered on it during extraction). **A constant column carries zero information** — a model can't split on it. Pure memory and compute overhead.

---

## ⚠️ 4. The re-run trap (important, and it bit us)

This line is **not safe to run twice**:
```python
df["onpromotion"] = df["onpromotion"] == "True"
```

- **First run:** strings `'True'`/`'False'` → real booleans. Correct.
- **Second run:** the column is already boolean. `True == "True"` evaluates to `False`. **Every value silently becomes False.**

No error is raised. The promotion signal is destroyed silently — the worst kind of bug.

### The lesson
Notebook cells that *mutate* `df` in place are re-run hazards. Make transformation cells **idempotent** — safe to run any number of times with identical results.

```python
df = df.drop(columns=["perishable"], errors="ignore")   # safe on repeat
```

**The reliable recovery habit:** reload from the last saved Parquet before re-running a transformation block. Cheap insurance.

**Why this got caught:** verifying the output (`value_counts()` showed 100% False) rather than assuming the cell worked. Always check the result of a mutation.

---

## 5. Grid reconstruction — the most important step in this phase

### The problem restated
The dataset only contains rows where a sale occurred. **No row = zero sales, but the model can't learn from rows that don't exist.** Train on this as-is and the model never sees a zero-demand day, so it systematically over-forecasts.

### Why a naive cross-join would be wrong

A full cross-join of 263 items × 54 stores × 956 days = 13.58 M rows. But two things make that incorrect:

**1. Not every item is sold at every store.**
```
Actual store-item pairs:  10,009
Theoretical max:          14,202
Pairs that exist:         70.5%
```
29.5% of combinations never existed — a store never carried that SKU. Filling zeros there **invents demand history for products that were never on the shelf.**

**2. Products are introduced and discontinued mid-timeline.**
```
store 1, item 108831:  2015-01-08 → 2017-02-11   (discontinued)
store 1, item 159156:  2015-01-02 → 2017-08-15   (still active)
```
Filling zeros past 2017-02-11 for item 108831 teaches the model **"demand collapsed"** when in reality the item simply left the assortment.

### The correct approach: bounded per-pair fill

Generate zeros **only between each pair's own first and last observed sale**.

```python
pair_range = (
    df.groupby(["store_nbr", "item_nbr"], observed=True)["date"]
      .agg(["min", "max"])
      .reset_index()
)

pair_range["date"] = pair_range.apply(
    lambda r: pd.date_range(r["min"], r["max"], freq="D"), axis=1
)
grid = pair_range.explode("date")[["store_nbr", "item_nbr", "date"]]
```

**Key mechanics:**
- `observed=True` — with categorical columns, this prevents pandas generating rows for combinations that never occurred
- `pd.date_range(min, max, freq="D")` — every daily date in that pair's active window
- `.apply(..., axis=1)` — runs row by row, producing a *list* of dates per pair
- **`.explode("date")`** — unpacks each list into individual rows. One pair with 900 dates becomes 900 rows
- Dtypes must be re-cast afterward — these operations reset them to int64

**Closest SQL analogue:** a cross-join to a calendar table, but bounded per pair rather than global.

### The result

| Metric | Value |
|---|---|
| Naive full grid | 13,577,112 |
| **Bounded grid** | **7,732,111** |
| Original sales rows | 5,934,528 |
| Implicit zeros added | 1,797,583 |

The bounded approach avoided **~5.8 M rows of fabricated history**.

### Filling and rejoining

```python
full = grid.merge(
    df[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]],
    on=["date", "store_nbr", "item_nbr"],
    how="left"
)
full["unit_sales"]  = full["unit_sales"].fillna(0).astype("float32")
full["onpromotion"] = full["onpromotion"].fillna(False).astype(bool)
```

`how="left"` = `LEFT JOIN`. Grid on the left, so every grid row survives; unmatched rows get NaN, which `fillna(0)` converts to real zeros.

Item and store attributes are rejoined afterward via `.drop_duplicates()` dimension tables — cheaper than carrying them through the explode.

---

## 6. Validation

```
Zero-sales rows: 1,797,881
Predicted:       1,797,583
Difference:      298   ← exactly the clipped negatives, now sitting at zero
Nulls remaining: 0
```

The 298 discrepancy **reconciles exactly** with the clipped negatives. That's a clean audit trail, not an unexplained gap.

### Intermittency by family

| Family | % zero-demand days |
|---|---|
| SEAFOOD | 28.7 |
| POULTRY | 26.7 |
| PREPARED FOODS | 25.3 |
| DELI | 22.2 |
| MEATS | 20.8 |

SEAFOOD is most intermittent — only 8 items, slow-moving.

### The headline number

Mean daily units, with zeros vs. sales-days-only:

| Family | With zeros | Sales days only | Over-forecast if uncorrected |
|---|---|---|---|
| POULTRY | 12.25 | 16.70 | **+36%** |
| MEATS | 9.77 | 12.34 | +26% |
| PREPARED FOODS | 8.17 | 10.94 | +34% |
| SEAFOOD | 5.33 | 7.47 | +40% |
| DELI | 4.62 | 5.94 | +29% |

**This gap is exactly the bias the model would have carried without grid reconstruction.** On perishable protein, a 36% over-forecast translates directly into spoilage.

---

## 7. Key takeaways

- *"The dataset only recorded days with sales, so 23% of the true series was implicitly missing. I reconstructed the full daily grid — but bounded per store-item pair by first and last observed sale, so I wasn't fabricating history for products a store never carried or had discontinued."*
- *"Quantifying it: uncorrected, poultry would have been forecast at 16.7 units/day against a true mean of 12.25 — a 36% over-forecast, which on perishable protein is direct spoilage cost."*
- *"I cut two years of history because promotion tracking only began mid-2014. Promotion is a primary demand driver, and imputing it would have taught the model that promotions don't affect sales."*
- *"Dtype optimization cut memory 75%, from 745 MB to 187 MB, which is what made the grid reconstruction feasible on an 8 GB machine."*

---

## Current pipeline state

```
data/raw/train.csv                    4.7 GB   (source, never touched after extraction)
  ↓ DuckDB extract, filtered to 5 protein families
data/processed/protein_sales.parquet  9.32 M rows
  ↓ date cut, clip negatives, dtype optimization
data/processed/protein_sales_clean.parquet  5.93 M rows, 187 MB
  ↓ bounded grid reconstruction + zero fill
data/processed/protein_grid.parquet   7.73 M rows, 184 MB  ← current working file
```

**Next phase:** feature engineering — lag features, rolling windows, calendar/holiday flags. This is where forecasting accuracy is actually won, and it's the phase that most depends on the continuous daily series built here.
