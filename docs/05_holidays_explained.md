# Phase 5 — Holiday Features: What We Did and Why

**Notebook:** `notebooks/05_holidays.ipynb`
**Input:** `protein_features.parquet` (7.46 M rows, 36 columns)
**Output:** `protein_features_hol.parquet` (7.46 M rows, 45 columns)

> **Verdict up front:** these features produced only a marginal improvement in the final model (see Phase 6). The engineering was correct; the signal was weak. That negative result is documented here rather than hidden.

---

## Why this was not a simple merge

`holidays_events.csv` is 350 rows and 6 columns, but joining it correctly required handling four separate traps. Each one was verified against the actual data before writing any join logic.

| Column | Values |
|---|---|
| `date` | the date |
| `type` | Holiday, Transfer, Additional, Bridge, Work Day, Event |
| `locale` | National, Regional, Local |
| `locale_name` | "Ecuador", a state name, or a city name |
| `description` | e.g. "Navidad", "Terremoto Manabi" |
| `transferred` | True / False |

---

## Trap 1: the join key changes per row

`locale` determines what `locale_name` matches against:

| locale | Applies to | Join key |
|---|---|---|
| National | all 54 stores | `date` |
| Regional | stores in that state | `date` + `state` |
| Local | stores in that city | `date` + `city` |

This is **three separate joins**, not one. The `city` and `state` columns retained from the store dimension in Phase 1 are what make it possible.

### Verifying the keys match before joining

```python
print(sorted(hol[hol["locale"]=="Regional"]["locale_name"].unique()))
print(sorted(df["state"].unique().tolist()))
```

**This check is the one people skip.** If the holiday file spelled a state differently — an accent, an abbreviation, different casing — the join would match nothing and silently produce a feature that is always False. No error, just a dead column.

In this case all four regional names (`Cotopaxi`, `Imbabura`, `Santa Elena`, `Santo Domingo de los Tsachilas`) matched the store dimension exactly.

**Always compare the value sets before a merge on text keys.**

---

## Trap 2: `transferred = True` means it was NOT a holiday that day

Ecuador officially moves some holidays to create long weekends.

- A row with `transferred = True` marks the **original** date — which was worked normally
- A separate row with `type = 'Transfer'` marks the date it actually moved to

```python
hol = hol[hol["transferred"] == False]     # drop the originals
# 'Transfer' rows are KEPT — they mark real holidays
```

Joining naively would flag the wrong day and miss the real one. 12 rows affected.

---

## Trap 3: `type = 'Work Day'` is the inverse of a holiday

A Work Day marks a Saturday that was *worked* to compensate for a bridge day. Treating it as a holiday flips the signal entirely. Separated into its own feature.

---

## Trap 4: row explosion

Multiple holidays can fall on the same date for the same locale — a national holiday plus a local festival.

```
Date+locale combos with >1 holiday: 7
  2016-05-01  National  Ecuador   2
  2016-05-07  National  Ecuador   2
  2016-05-08  National  Ecuador   2
  2016-07-24  Local     Guayaquil 2
```

Without deduplication, **every sales row on those dates would double** — 7.46 M rows silently becoming more, with double-counted demand. This produces no error.

### Two defenses

```python
nat = hol[hol["locale"]=="National"][["date","type"]].drop_duplicates(subset=["date"])
```

```python
assert len(df) == before, "ROW EXPLOSION — dedup failed"
```

The `assert` is the important habit: it halts execution immediately rather than letting corrupted data flow downstream. **Use an assert on any merge where explosion is possible.** It is the cheapest insurance in data engineering.

---

## The join result

```
National holiday dates:        46
Regional (date, state) pairs:  10
Local (date, city) pairs:      64

Rows before: 7,459,611 | after: 7,459,611  ✓

Coverage:  National 4.8% | Regional 0.0% | Local 0.4% | Any 5.2%
```

### Investigating "Regional 0.0%"

0.0% is suspicious enough to check rather than assume — it's exactly the silent-mismatch failure mode described above.

```
Regional holiday rows: 2,207   ← not zero, just rounds to 0.0%
```

Confirmed legitimate: the four regional states contain only 7 of 54 stores, across 10 holiday dates (provincialization anniversaries recurring annually). Small by construction.

**Lesson:** when a coverage percentage looks wrong, check the raw count before assuming a bug.

---

## Proximity features

```python
idx = np.searchsorted(hol_array, row_dates, side="left")
```

`np.searchsorted` binary-searches a sorted array to find where each date would slot in — O(log n) instead of comparing every date to every holiday. On 7.46 M rows this is the difference between minutes and hours.

### The edge-case bug (worth remembering)

The first implementation of `days_from_prev_holiday` was **silently wrong**. For dates falling before a store's first holiday, `searchsorted(...) - 1` returns `-1`; `np.clip(idx, 0, ...)` forced it to index 0 — the *first* holiday, which is in the future. Subtracting a future date gave a negative, which the outer clip floored to 0.

Result: `days_from_holiday` was stuck at 0 at the start of every store's series. No error raised.

```python
no_prev = idx < 0
days[no_prev] = 30      # explicit handling of "no previous holiday"
```

**This was caught only by inspecting the output.** The countdown feature looked correct, so it would have been easy to assume the companion feature was fine too.

### Verified correct after the fix
```
2015-02-14   to=2   from=30
2015-02-16   to=0   from=0     ← holiday
2015-02-18   to=30  from=1
2015-02-19   to=30  from=2
```

---

## Event features

The earthquake appears as 31 consecutive daily rows (2016-04-16 → 2016-05-16), tagged `Terremoto Manabi+0` through `+30`.

### Why flag rather than delete

Dropping 31 days would break the continuous daily series and corrupt every lag computed across the gap. Flagging isolates the anomaly while keeping the timeline intact.

**Flag, don't delete** — the standard outlier approach in time series.

---

## Validating the hypotheses

### Earthquake — dilution effect

| Scope | During | Normal | Difference |
|---|---|---|---|
| All stores | 7.63 | 7.81 | **−2.3%** |
| Manabi only | 5.25 | 4.42 | **+18.9%** |

The national average shows nothing. **Manabi — the affected province, 3 of 54 stores — shows a +18.9% demand shock.**

**Lesson: check the affected subgroup, not just the aggregate.** A real regional effect was completely invisible at the national level.

### Holiday proximity — hypothesis not supported

The expectation was a pre-holiday stock-up ramp: demand climbing as `days_to_holiday` approaches 0.

```
days_to_holiday:  0     1     2     3     4     5     6     7
mean unit_sales: 8.34  7.58  7.77  7.42  6.82  8.01  7.92  8.45
```

No ramp. Day 7 is the highest value. Controlling for day-of-week (Saturdays only) did not change the conclusion.

### Holiday flag itself — modest real effect

```
Holiday:     8.36
Non-holiday: 7.77   (+7.6%)
```

---

## Key takeaways

- *"Joining the holiday calendar required three separate joins on different keys — national on date, regional on date and state, local on date and city — plus handling transferred holidays, which mark dates that were worked normally, and 'Work Day' entries which are the inverse of a holiday."*
- *"I put an assert on the row count after the merge. Multiple holidays can land on the same date for the same locale, and without deduplication every sales row on those dates would silently double."*
- *"The earthquake showed −2.3% nationally but +18.9% in the affected province. Aggregate statistics hid a real regional shock — I only found it by checking the subgroup."*
- *"I hypothesised a pre-holiday demand ramp for perishables. The data didn't support it at store-item-day granularity, and the model later confirmed the proximity features contributed almost nothing. I kept them because they marginally reduced large errors, but the honest result is that the hypothesis was wrong."*
- *"One proximity feature was silently wrong at the start of each store's series due to an unhandled index edge case. It produced no error — I found it only by inspecting the output rather than trusting that it worked."*
