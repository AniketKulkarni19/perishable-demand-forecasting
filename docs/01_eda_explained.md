# Phase 1 — Exploratory Data Analysis: What We Did and Why

**Notebook:** `notebooks/01_eda.ipynb`
**Goal:** Understand the raw data well enough to make scoping and cleaning decisions with evidence rather than guesswork.

---

## 1. Environment setup

### What we did
Built an isolated Python 3.12 environment using `uv`, rather than using the system Python (3.14).

### Why
Python 3.14 was newly released. Key ML packages (LightGBM, SHAP, Prophet) often lack prebuilt binaries ("wheels") for the newest Python version, which forces pip to compile from source — a common source of multi-hour dependency failures, especially on Intel Macs. Python 3.12 has the widest, most stable wheel coverage.

**Key point:** `uv` manages both the Python version *and* the virtual environment. The `uv.lock` file records exact versions of all 110+ packages, so the environment is reproducible on any machine. This replaces the older `requirements.txt` approach.

### Commands worth remembering
```bash
uv venv                    # create isolated environment
uv add <package>           # install + record in pyproject.toml
uv run jupyter lab         # run a command inside the environment
```

---

## 2. The core memory problem

### The situation
`train.csv` is **4.7 GB / ~125 million rows**. The machine has 8 GB RAM.

### Why `pd.read_csv()` would fail
Pandas needs roughly 2–3× a file's size in RAM to load and process it — so ~10–14 GB for this file. The machine would swap to disk, crawl, and likely crash the kernel.

This is the single most common wall people hit on real datasets. Most tutorials avoid it by using small toy data or beefy cloud machines.

### The solution: DuckDB (out-of-core processing)
DuckDB reads the CSV **on disk**, runs SQL against it, and returns only the result set to Python. The 125M rows never enter RAM.

```python
con = duckdb.connect()
result = con.execute("""
    SELECT family, COUNT(*)
    FROM read_csv_auto('train.csv')
    GROUP BY family
""").df()
```

**Why it's fast** (30 seconds on a 1.4 GHz i5): DuckDB is *columnar* and *vectorized*. It only reads the columns the query touches and processes them in batches, skipping the rest of each row entirely. Pandas would parse all 4.7 GB into Python objects first.

**Mental model:** think of DuckDB as a bouncer at the door. The full dataset stays outside in the file; only the rows you explicitly asked for are allowed into memory.

---

## 3. Dataset structure

Classic **star schema** — one fact table, several dimension tables:

| File | Role | Size | Key columns |
|---|---|---|---|
| `train.csv` | Fact (sales) | 4.7 GB | date, store_nbr, item_nbr, unit_sales, onpromotion |
| `items.csv` | Dimension | 99 KB | item_nbr, family, class, **perishable** |
| `stores.csv` | Dimension | 1.4 KB | store_nbr, city, state, type, cluster |
| `oil.csv` | Context | 20 KB | date, oil price (Ecuador's economy is oil-dependent) |
| `holidays_events.csv` | Context | 22 KB | date, holiday type, locale |
| `transactions.csv` | Context | 1.5 MB | date, store_nbr, total transactions |

`items.csv` carries a **`perishable` flag (1/0)** supplied by the retailer — we didn't have to infer which products spoil. This is the column that makes this dataset the right fit for a perishable-goods forecasting project.

---

## 4. Scoping decision

### The finding
986 of 4,100 items are flagged perishable, spanning 9 families:

| Family | Items | Rows in full dataset |
|---|---|---|
| DAIRY | 242 | 8.99 M |
| PRODUCE | 306 | 7.15 M |
| BREAD/BAKERY | 134 | 4.66 M |
| DELI | 91 | 4.12 M |
| MEATS | 84 | 2.43 M |
| POULTRY | 54 | 1.74 M |
| EGGS | 41 | 1.58 M |
| PREPARED FOODS | 26 | 0.77 M |
| SEAFOOD | 8 | 0.26 M |

### The decision: 5 protein families (V1)
**MEATS, POULTRY, DELI, PREPARED FOODS, SEAFOOD** — 9.3 M rows, 263 items, 54 stores, 4.5 years.

**Two reasons:**

1. **Iteration speed.** After feature engineering the frame grows from ~6 to ~40 columns. At 9.3 M rows that's ~1.5 GB after dtype optimization — comfortable. At 32 M rows (all 9 families) it would be ~5 GB — right at the edge of 8 GB RAM, meaning swapping, occasional kernel crashes, and ~3.5× longer training runs. During a learning phase with dozens of experiments, that friction compounds badly.

2. **Domain narrative.** Protein categories align with four years of meat-processing industry experience. That's a differentiator no generic portfolio project has.

### The hybrid approach
- **EDA runs on 100% of the data** (all 125 M rows) via DuckDB aggregations — the claim of full-dataset analysis is literally true.
- **Modeling runs on the scoped subset** for tractable iteration.

This is how practitioners actually work on laptops. It's a strength to state explicitly, not a limitation to hide.

### Why it's "parameterized"
The family list lives in a single variable:
```python
FAMILIES = ['MEATS', 'POULTRY', 'DELI', 'PREPARED FOODS', 'SEAFOOD']
```
V2 (all 9 families) means changing that one line and re-running. Nothing downstream knows or cares which families it received. Same principle as a stored procedure taking a parameter instead of a hardcoded filter.

---

## 5. Extraction strategy

Three-table join filtered to the target families, written **directly to Parquet**:

```python
con.execute(f"COPY ({extract_query}) TO 'protein_sales.parquet' (FORMAT PARQUET)")
```

**Why `COPY ... TO` matters:** DuckDB streams query results straight to disk. The 9.3 M rows never fully materialize in Python memory. This is the difference between "works on 8 GB" and "kernel died."

**Why Parquet over CSV:**
- Columnar and compressed — 4.7 GB of CSV becomes a few hundred MB
- **Preserves dtypes** — a CSV would lose all the type optimization on every reload
- Reads in seconds, so the 4.7 GB source file never needs to be touched again

---

## 6. Four findings that shaped everything downstream

### Finding 1: No zero-sales rows exist
```
Zero unit_sales rows: 0
```
Not a single one. The dataset **only records days where a sale occurred**. If a store sold no ground beef on a Tuesday, there is simply no row.

**Grid math:** 263 items × 54 stores × 1,684 days = 23.9 M possible rows. Only 9.3 M exist → **61% of the grid is missing**, almost all genuine zero-demand days.

**Consequence:** training on this as-is means the model never sees a zero-demand day and will systematically **over-forecast**. This became the biggest single cleaning task (see Phase 2).

### Finding 2: `onpromotion` is 21.3% null — and it's purely historical

| Year | % missing |
|---|---|
| 2013 | 100.0 |
| 2014 | 23.9 |
| 2015 | 0.0 |
| 2016 | 0.0 |
| 2017 | 0.0 |

The retailer started tracking promotions mid-2014. This is a clean structural boundary, not scattered missingness — which makes it a defensible cut point rather than an imputation problem.

### Finding 3: 429 negative `unit_sales` values
Returns and adjustments. Only 0.005% of rows, but they're not valid demand.

### Finding 4: `unit_sales` is fractional (e.g. `29.904`)
Not an error. Some products sell **by weight** — kilograms of chicken, not units. Expected in protein categories.

---

## 7. pandas ↔ SQL translation reference

| SSMS | pandas |
|---|---|
| `WHERE col = x` | `df[df["col"] == x]` |
| `GROUP BY col` | `.groupby("col")` |
| `COUNT(*)` | `.size()` |
| `COUNT(DISTINCT col)` | `.nunique()` |
| `SELECT DISTINCT col` | `.unique()` |
| `ORDER BY col DESC` | `.sort_values("col", ascending=False)` |
| `SELECT col1, col2` | `df[["col1", "col2"]]` |
| `JOIN` | `.merge()` |
| `LEFT JOIN` | `.merge(..., how="left")` |
| `YEAR(date)` | `df["date"].dt.year` |
| `COUNT(*) WHERE col IS NULL` | `df["col"].isna().sum()` |
| `CASE WHEN x < 0 THEN 0 ELSE x END` | `df["x"].clip(lower=0)` |

**Where the analogy breaks down:** time-series feature engineering (lags, rolling windows) maps to SQL *window functions* — `LAG()` and `OVER (PARTITION BY ... ORDER BY ...)`. In pandas these are `.shift()` and `.rolling()`, applied within `.groupby()`. Covered in Phase 3.

**Boolean filtering** is the pattern to internalize:
```python
df[df["perishable"] == 1]
```
The inner expression produces True/False for every row; wrapping it in `df[...]` keeps only the True rows. It's `WHERE`, but the condition is a first-class object you can store, combine, and reuse.

---

## 8. Talking points for interviews

- *"The raw dataset was 125 million rows against 8 GB of RAM, so I used DuckDB for out-of-core processing — aggregations ran across the full dataset without ever loading it into memory."*
- *"I discovered the dataset only recorded non-zero sales days, meaning 61% of the item-store-day grid was implicitly missing. Left uncorrected, that would have biased every forecast upward."*
- *"I scoped modeling to protein categories for iteration speed, but built the pipeline parameterized by family so it scales to the full perishable catalog by changing one variable."*
