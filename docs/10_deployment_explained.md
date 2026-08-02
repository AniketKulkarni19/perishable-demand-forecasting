# Phase 10–12 — Productionization: Refactor, API, Container, Deployment

**Files:** `src/perishable_demand_forecasting/`, `app/main.py`, `Dockerfile`, `streamlit_app.py`

> This is the phase most portfolio projects skip. It's the difference between "I analyzed some data" and "I built something that runs."

---

## Phase 10 — Refactoring notebooks into a package

### Why this had to happen first

**An API cannot import from a notebook.** All feature logic lived in `.ipynb` cells, which are JSON files, not importable Python.

There's a second, subtler reason: if the API reimplemented feature engineering separately, the two implementations would eventually diverge — and the API would silently compute features differently than training did. Same code path for training and serving is the only way to guarantee they agree.

### What was built

```
src/perishable_demand_forecasting/
    __init__.py
    features.py     # build_features() — the full pipeline
    predict.py      # load_model(), predict()
```

The logic is identical to the notebooks. What changed: named functions, docstrings, type hints, and a single entry point (`build_features()`) that the API calls instead of reimplementing 40 lines.

### Two details worth remembering

**Path resolution relative to the file, not the working directory:**
```python
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lgbm_holidays.txt"
```
`Path(__file__)` is *this module's* location. Hardcoding `'../models/...'` would break the moment the working directory changed — which it does when the API runs, when Docker runs, and when Streamlit runs. Three different working directories, one correct path.

**Fail loudly on bad input:**
```python
missing = [f for f in feature_names if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")
```
When an API receives a malformed request, you want a clear error naming what's absent — not an opaque failure deep inside LightGBM.

### Validation
The refactored code was verified against the notebook implementation using the same leakage check from Phase 3:
```
Store-item pairs in sample: 425
lag_1 nulls:               425
Match: True
```
One null per pair means groupby boundaries are respected — identical behaviour to the original.

### The reload gotcha
```python
import importlib
importlib.reload(P)
```
A Jupyter kernel caches imported modules. Without an explicit reload, edits to a `.py` file won't be picked up — you'll be testing stale code and wondering why your fix didn't work.

---

## Phase 11 — FastAPI service

### What deployment actually means

Before: the model is a file that runs when *you* open a notebook. Nobody else can use it; nothing else can call it.

After: a **service** — something running on a computer that isn't yours, that anything can send a request to and get a prediction back.

Four layers:

| Layer | What it is | Ours |
|---|---|---|
| Model | Learned rules, inert on its own | `lgbm_holidays.txt` |
| API | Wraps the model, speaks HTTP | `app/main.py` (FastAPI) |
| Container | Code + runtime + dependencies, frozen together | `Dockerfile` |
| Host | A computer on the internet running it 24/7 | Streamlit Cloud |

### The design problem: where do historical features come from?

The model needs 43 features, most of them **historical** — `lag_7`, `roll_mean_28`, `days_since_sale`. An API caller can't supply those; they'd need the item's full sales history.

| Option | Trade-off |
|---|---|
| Caller sends 28 days of history | Most honest, clunky payload |
| **API looks up history internally** ✓ | Clean interface, ships a data file |
| Serve precomputed forecasts | Closest to production, arguably not "inference" |

Chose internal lookup — clean interface, genuinely demoable, and the required slice is small (124K rows, 16.6 MB).

### What FastAPI gives you for free

```python
class ForecastRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, le=54)
    item_nbr: int
    forecast_date: date
    apply_cost_bias: bool = False
```

Defining this Pydantic model produces:
- **Automatic validation** — `ge=1, le=54` rejects invalid store numbers before your code runs
- **Interactive documentation** at `/docs` — a live page where anyone can send real requests without writing curl commands
- **Type coercion** — the date string becomes a `date` object automatically

### Structural choices

**`@app.on_event("startup")`** loads the model and data **once** when the service boots, not on every request. Loading a model per-request would make the API unusably slow.

**A `/health` endpoint** is what hosting platforms ping to check the service is alive. Standard practice, and platforms expect it.

**Meaningful 404s:**
```python
detail=f"No data for store {req.store_nbr}, item {req.item_nbr} on {req.forecast_date}. Try /items for valid combinations."
```
An error message that tells the caller how to fix their request.

---

## Phase 12 — Docker

### Why containers exist

"Works on my machine" is a real problem. The server won't have your Homebrew `libomp`, your exact numpy version, or your Python 3.12 install.

A container packages **the code, the runtime, and every dependency** into one artifact that behaves identically anywhere Docker runs. Shipping the whole kitchen, not just the recipe.

### The Dockerfile, annotated

```dockerfile
FROM python:3.12-slim              # minimal base with the right Python

RUN apt-get install -y libgomp1    # Linux equivalent of macOS libomp
                                   # LightGBM fails at import without it

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project    # dependencies ONLY

COPY src/ app/ models/ data/serving/ ./               # then the code
RUN uv sync --frozen --no-dev                         # then the project

CMD uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Layer caching is why dependencies come first.** Docker caches each instruction. Since `pyproject.toml` changes rarely, editing `main.py` skips the slow dependency reinstall entirely — 14 seconds saved on every rebuild.

**`--host 0.0.0.0` is critical.** The default `127.0.0.1` only accepts connections from *inside* the container, making the service unreachable from outside. This is a classic first-container mistake.

### Two build failures, both instructive

**1. Chicken-and-egg with the project install**
```
× Failed to build `perishable-demand-forecasting`
╰─▶ Expected a Python module at: src/perishable_demand_forecasting/__init__.py
```
`pyproject.toml` declares the project as an installable package, so `uv sync` tried to build it — but `src/` was copied *after* that step.

Fix: `--no-install-project` for the dependency layer, then a second `uv sync` after the code lands. Preserves caching *and* works.

**2. Missing README**
```
╰─▶ failed to open file `/app/README.md`: No such file or directory
```
`pyproject.toml` references README.md as the package long-description. Copied it alongside `pyproject.toml` to keep it in the cached layer.

**General lesson:** Docker build errors are usually about *what exists at that point in the build*, not about your code being wrong. Read them as "at this step, this file wasn't there yet."

### Result
```
perishable-forecast:latest   1.29 GB disk   298 MB compressed
```

---

## Deployment: what actually happened

### Hugging Face Spaces — no longer viable

Docker and Gradio SDKs both now require a paid tier; only Static is free. Plans change; verify current pricing rather than trusting a guide.

### Streamlit Community Cloud — free, no card

<https://perishable-demand-forecasting-yqyz9zztek2qmerncudrxf.streamlit.app/>

Limits: ~1 GB memory, apps sleep after 12 hours idle, public repos only. All fine for this.

### ⚠️ The pandas 3.0 problem — and knowing when to stop

Streamlit crashed on **every dropdown change**, locally. The process died silently, with no traceback — a killed process, not a Python exception.

**Systematic bisection:**

| Test | Result | Eliminates |
|---|---|---|
| Prediction path in a plain script | ✅ Works | Model, data, LightGBM |
| Stop the Docker container (freeing 1.5 GB) | ❌ Still crashes | Memory pressure |
| Minimal app — no caching, no charts | ❌ Still crashes | Caching, chart rendering |
| `OMP_NUM_THREADS=1` | ❌ Still crashes | OpenMP threading |
| Precomputed forecasts (no LightGBM at all) | ❌ Still crashes | LightGBM entirely |
| **Streamlit with no pandas** | ✅ **Works** | → **pandas 3.0 is the culprit** |

**The cost of running bleeding-edge versions.** pandas 3.0 was the right call for the pipeline, but it bit twice:
1. **Blocked SHAP** — numba caps at numpy <2.3; pandas 3.0 requires numpy ≥2.5. Irreconcilable
2. **Broke Streamlit** — the library hasn't caught up to pandas 3.0

**The fix:** pin only the *deployment* environment. Streamlit Cloud builds a fresh Linux environment from `requirements.txt`:
```
streamlit
pandas<3
numpy<2.3
pyarrow
```
Notebooks keep pandas 3.0 locally; the deployed app uses older versions. The app only does `read_parquet`, filtering, and a chart — nothing that needs 3.0.

**And it worked.** The deployed version handles dropdown changes fine — the crash was a local environment problem that never existed on the target platform.

**The judgment call worth internalizing:** after five bisection steps pointing at a macOS-specific library conflict, the right move was to stop debugging locally and let the cloud build decide. Recognising when a dependency fight isn't worth having — and when the target environment differs enough that local reproduction doesn't matter — is a real engineering skill.

---

## Two serving paths, deliberately

| Path | Implementation | Reflects |
|---|---|---|
| **Real-time inference** | `app/main.py` — FastAPI + Docker | On-demand prediction, containerized |
| **Batch precomputed** | `scripts_precompute.py` → Streamlit | How this runs in production |

**The honest framing:** a real retailer would run demand forecasting as a **scheduled batch job** — every night, forecast everything, write to a table. A live REST API is arguably the *less* realistic architecture for this use case.

The API demonstrates real-time serving capability. The batch demo reflects the actual production pattern. Both are in the repo, and knowing the difference matters more than picking one.

---

## Key takeaways

- *"I refactored the feature engineering out of notebooks into an importable package, so training and serving share one implementation. If the API reimplemented features separately, the two would eventually diverge and predictions would silently break."*
- *"Docker matters because the server won't have my Homebrew OpenMP install or my exact package versions. The container packages the runtime and dependencies together so it behaves identically anywhere."*
- *"Streamlit crashed on every interaction locally. I bisected it — the model worked in a plain script, it wasn't memory, wasn't threading, wasn't caching, and it still crashed with LightGBM removed entirely. Streamlit with no pandas worked, which isolated it to pandas 3.0. Rather than keep debugging a macOS-specific conflict, I pinned older versions in requirements.txt and let the Linux cloud build resolve it. It deployed cleanly."*
- *"Running bleeding-edge versions has a cost. pandas 3.0 was right for the pipeline but blocked SHAP through a numpy conflict and broke Streamlit. The fix was pinning only the deployment environment, not downgrading the whole project."*
- *"I built both a real-time API and a batch-precomputed demo, because production demand forecasting actually runs as a nightly batch job. The API demonstrates serving capability; the batch path reflects the real architecture."*
