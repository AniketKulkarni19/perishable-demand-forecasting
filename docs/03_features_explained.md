# Phase 3 — Feature Engineering: What We Did and Why

**Notebook:** `notebooks/03_features.ipynb`
**Input:** `data/processed/protein_grid.parquet` (7.73 M rows, 11 columns)
**Output:** `data/processed/protein_features.parquet` (7.46 M rows, 36 columns)

---

## The one concept that governs this entire phase: leakage

A forecasting model predicts tomorrow using **only what is knowable today**.

If any feature contains information from the target day or later, the model scores brilliantly in validation and fails completely in production. This is the number one way time-series projects go wrong, and it produces **no error message** — the code runs fine, the metrics look great, and the model is worthless.

**Concrete example:** to predict sales on day T, a 7-day rolling average must cover days T−7 through T−1. If it includes day T, the model is being handed a piece of the answer.

### Two rules applied to every historical feature

1. **Shift before rolling.** Every lag and rolling statistic is computed on a series that has already been shifted by 1 day.
2. **Always group by store-item.** Every `.shift()` and `.rolling()` happens *within* `groupby(["store_nbr", "item_nbr"])`, so one item's history never bleeds into the next item's early rows.

Without the groupby, the last rows of one store-item pair silently become the first lag values of the next pair. Another failure with no error message.

---

## 0. Sorting is mandatory

```python
df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
```

`.shift()` and `.rolling()` operate on **row order**, not on date values. They have no idea what a date is. Unsorted data produces silently wrong lags.

**Sort before any time-based operation. Every time.**

---

## 1. Calendar features

No leakage risk — the calendar is known arbitrarily far in advance.

```python
d = df["date"].dt
df["dayofweek"]  = d.dayofweek.astype("int8")     # 0 = Monday
df["month"]      = d.month.astype("int8")
df["is_weekend"] = (d.dayofweek >= 5)
df["is_payday"]  = ((d.day == 15) | (d.is_month_end))
```

`.dt` is the datetime accessor — the pandas equivalent of T-SQL's `DATEPART()`. Assigning it to `d` once avoids repeating `df["date"].dt` on every line.

### The domain feature: `is_payday`

Ecuadorian public-sector wages are paid on the **15th and the last day of the month**. Grocery demand spikes on and immediately after those dates. Favorita's own competition documentation flags this as a demand driver.

This is the kind of feature that comes from understanding the business, not from a tutorial — worth calling out explicitly.

---

## 2. Lag features

```python
grp = df.groupby(["store_nbr", "item_nbr"], observed=True)["unit_sales"]
LAGS = [1, 7, 14, 28]
for lag in LAGS:
    df[f"lag_{lag}"] = grp.shift(lag).astype("float32")
```

Grouping once and reusing `grp` matters — grouping 7.7 M rows repeatedly is expensive.

### Why these specific lags

| Lag | Captures |
|---|---|
| **1** | Yesterday — immediate momentum |
| **7** | Same weekday last week — grocery demand is strongly weekly (Saturday ≠ Tuesday) |
| **14** | Two weeks back, same weekday — confirms whether the weekly pattern is stable |
| **28** | Four weeks back — roughly monthly, aligns with pay cycles |

Note that 7, 14, and 28 are all multiples of 7. **This is deliberate**: it preserves weekday alignment. Comparing a Monday to a previous Monday is far more informative than comparing it to a Sunday.

### SQL equivalent
```sql
LAG(unit_sales, 7) OVER (PARTITION BY store_nbr, item_nbr ORDER BY date)
```
Nearly identical concept.

### Validation that it worked
```
Nulls in lag_1:  10,009   ← exactly one per store-item pair
Nulls in lag_28: 272,500
```
`lag_1` nulls matching the pair count exactly is proof that the groupby boundary was respected. **Always check this** — it's the cheapest possible test that lags didn't leak across groups.

---

## 3. Rolling window features

```python
shifted = grp.shift(1)          # ← THE CRITICAL LINE

for w in [7, 14, 28]:
    roll = shifted.groupby([df["store_nbr"], df["item_nbr"]], observed=True).rolling(w, min_periods=1)
    df[f"roll_mean_{w}"] = roll.mean().reset_index(level=[0,1], drop=True).astype("float32")
    df[f"roll_std_{w}"]  = roll.std().reset_index(level=[0,1], drop=True).astype("float32")
    df[f"roll_max_{w}"]  = roll.max().reset_index(level=[0,1], drop=True).astype("float32")
```

### Why `shifted = grp.shift(1)` comes first

The entire series is moved back one day **before** any window is computed. So the 7-day mean for day T covers T−7 through T−1 — **day T is excluded**.

Rolling on the raw series would include the target value in its own predictor. The model would score beautifully and be useless.

### Why three statistics per window

| Statistic | Signal |
|---|---|
| `mean` | Typical demand level |
| `std` | Volatility — a high-variance item needs more safety stock |
| `max` | Recent peak — useful for detecting promotional spikes |

### The awkward index handling
`.rolling()` after `.groupby()` returns a MultiIndex (group keys + original index). `.reset_index(level=[0,1], drop=True)` strips the group levels so the result realigns with `df`. Fiddly, but required.

`min_periods=1` computes a value even when fewer than `w` rows are available, so early rows get partial windows instead of nulls.

### SQL equivalent
```sql
AVG(unit_sales) OVER (
    PARTITION BY store_nbr, item_nbr
    ORDER BY date
    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
)
```
Note **`1 PRECEDING`**, not `CURRENT ROW`. That's the same leakage guard.

### Manual verification (worth doing once)
```
Row 3 (2015-01-06): roll_mean_7 = 1.6667
Rows 0-2 unit_sales: 1.0, 1.0, 3.0  →  5/3 = 1.6667  ✓
Row 3's own value (2.0) is excluded  ✓
```
Hand-checking one window is the only way to be certain the shift is right.

---

## 4. Promotion features

```python
df["promo_lag_1"]    = promo_grp.shift(1)              # on promo yesterday?
df["promo_count_28"] = ...rolling(28).sum()            # promo days in trailing 4 weeks
```

**Why lagged promotion matters:** promotions cluster in multi-day blocks and have *aftereffects*. A promotion pulls demand forward, so the days following a promo often dip below normal. Recent promo intensity helps the model handle that rebound.

**Note:** `onpromotion` for day T is legitimately known in advance — retailers plan promotions — so the raw column stays as a feature. The historical versions are still shifted for consistency.

---

## 5. Intermittency features

These exist because 23% of the reconstructed series is zero-demand days.

### `days_since_sale`
```python
sold = (df["unit_sales"] > 0)
df["days_since_sale"] = (
    df.assign(_s=sold)
      .groupby(["store_nbr", "item_nbr"], observed=True)["_s"]
      .transform(lambda s: s.shift(1).groupby((s.shift(1) == True).cumsum()).cumcount())
)
```

The gnarliest code in the project. Read it as three steps:
1. **Shift** to avoid leakage
2. **`cumsum()` on the sale flag** creates a new group ID each time a sale occurs — a "group boundary" marker
3. **`cumcount()`** counts rows since that boundary

This is the standard pandas idiom for *"rows since last occurrence."* Understand what it outputs; don't worry about reproducing it from memory.

### `zero_rate_28`
Share of the trailing 28 days with zero sales. Distinguishes *"this pair is chronically sporadic"* from *"this is normally a daily seller currently in a lull."*

### Honest caveat
`days_since_sale` is only meaningful for genuinely intermittent items. For a pair that sells almost every day, it stays near zero and contributes nothing. It should earn its keep on SEAFOOD (28.7% zero days) and slow-moving SKUs.

**If SHAP later shows it contributes nothing, dropping it is a legitimate finding — not a failure.** Testing a hypothesis and reporting a negative result is good practice.

---

## 6. Handling nulls: drop the warm-up period

```python
df = df[df["lag_28"].notna()].copy()
```

### Why dropping is correct
Every pair's first 28 days **cannot** have a valid `lag_28`. That's structural, not dirty data.

| Option | Trade-off |
|---|---|
| **Drop warm-up** ✓ | Loses 272,500 rows (3.5%). Clean |
| Impute 0 or mean | Teaches the model false patterns during warm-up; muddies early-life item behaviour |

### Why filtering on `lag_28` alone is sufficient
It's the longest lookback. Any row with a valid `lag_28` necessarily has valid shorter lags and rolling windows. One condition covers all 16 null-bearing columns.

### Validation
```
Dropped: 272,500 rows (3.5%)
Remaining: 7,459,611
Nulls remaining: None
New date range: 2015-01-29 to 2017-08-15
```
The new start date is **exactly 28 days** after 2015-01-01. That confirms the warm-up logic did precisely what was intended.

---

## Feature inventory (36 columns)

| Group | Count | Columns |
|---|---|---|
| Identifiers | 3 | store_nbr, item_nbr, date |
| Target | 1 | unit_sales |
| Item/store attributes | 6 | family, class, city, state, store_type, cluster |
| Known-in-advance | 1 | onpromotion |
| Calendar | 8 | dayofweek, day, month, year, weekofyear, is_weekend, is_payday, days_in_month |
| Lags | 4 | lag_1, lag_7, lag_14, lag_28 |
| Rolling | 9 | roll_{mean,std,max}_{7,14,28} |
| Promotion history | 2 | promo_lag_1, promo_count_28 |
| Intermittency | 2 | days_since_sale, zero_rate_28 |

---

## Key takeaways

- *"The main risk in time-series feature engineering is leakage — any feature containing information from the target day makes the model look excellent in validation and useless in production. Every lag and rolling statistic was computed on a series shifted by one day first, so a 7-day mean for day T covers T−7 through T−1."*
- *"I validated the groupby boundaries by checking that lag_1 had exactly 10,009 nulls — one per store-item pair. If history had bled across pairs, that count would have been lower."*
- *"Lags at 7, 14, and 28 days are deliberate multiples of seven, preserving weekday alignment. Grocery demand is strongly weekly, so comparing a Monday to a previous Monday is far more informative than to the preceding day."*
- *"Because 23% of the series is zero-demand days, I added intermittency features — days since last sale and trailing zero rate — which are standard for slow-moving SKUs."*
