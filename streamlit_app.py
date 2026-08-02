"""Interactive demo for the perishable demand forecasting model.

Forecasts are precomputed in batch (see scripts_precompute.py), which mirrors
how demand forecasting runs in production: a scheduled job generates forecasts
nightly and serves them from a table. Real-time inference is implemented
separately in app/main.py as a FastAPI service.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Perishable Demand Forecasting", layout="centered")


@st.cache_data
def get_data():
    df = pd.read_parquet(ROOT / "data" / "serving" / "predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    for c in ["family", "city", "store_type"]:
        df[c] = df[c].astype(str)
    return df


data = get_data()

st.title("Perishable Demand Forecasting")
st.caption(
    "Daily unit-sales forecasts for fresh protein across 54 supermarkets. "
    "LightGBM trained on 7.5M rows of Corporación Favorita retail data."
)

c1, c2 = st.columns(2)
with c1:
    family = st.selectbox("Category", sorted(data["family"].unique()))
    fam = data[data["family"] == family]
    store = st.selectbox("Store", sorted(fam["store_nbr"].unique()))
with c2:
    sub = fam[fam["store_nbr"] == store]
    item = st.selectbox("Item", sorted(sub["item_nbr"].unique()))
    isub = sub[sub["item_nbr"] == item]
    fdate = st.selectbox("Date", sorted(isub["date"].dt.date.unique()))

bias = st.checkbox(
    "Apply cost-optimal bias (×0.80)",
    help="Scales forecasts down to minimize spoilage cost. Reduces accuracy "
         "but cuts total cost, since over-forecasting costs ~3x more than "
         "under-forecasting for perishable goods.",
)

row = isub[isub["date"].dt.date == fdate]

if not row.empty:
    col = "forecast_cost_optimal" if bias else "forecast"
    pred = float(row[col].iloc[0])
    actual = float(row["unit_sales"].iloc[0])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast", f"{pred:.2f} units")
    m2.metric("Actual", f"{actual:.2f} units")
    m3.metric("Error", f"{pred - actual:+.2f}")

    hist = isub.sort_values("date")[["date", "unit_sales", "forecast"]].set_index("date")
    st.line_chart(hist)

st.divider()
st.caption(
    "Test WAPE 41.70 vs 48.60 baseline · 38.3% reduction in forecast-related costs · "
    "[Source](https://github.com/AniketKulkarni19/perishable-demand-forecasting)"
)
