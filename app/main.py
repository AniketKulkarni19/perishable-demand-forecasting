"""FastAPI service for perishable demand forecasting."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perishable_demand_forecasting.predict import load_model, predict  # noqa: E402

app = FastAPI(
    title="Perishable Demand Forecasting",
    description="Forecasts daily unit sales for fresh protein categories. "
                "Optionally applies a cost-optimal bias that reduces spoilage "
                "cost at the expense of forecast accuracy.",
    version="1.0.0",
)

MODEL = None
DATA = None


@app.on_event("startup")
def startup():
    global MODEL, DATA
    MODEL = load_model()
    DATA = pd.read_parquet(ROOT / "data" / "serving" / "features.parquet")
    DATA["date"] = pd.to_datetime(DATA["date"])


class ForecastRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, le=54, description="Store number (1-54)")
    item_nbr: int = Field(..., description="Item number")
    forecast_date: date = Field(..., description="Date to forecast (2017-07-31 to 2017-08-15)")
    apply_cost_bias: bool = Field(False, description="Apply cost-optimal 0.80 scaling")


class ForecastResponse(BaseModel):
    store_nbr: int
    item_nbr: int
    forecast_date: date
    family: str
    predicted_units: float
    cost_bias_applied: bool


@app.get("/")
def root():
    return {"service": "Perishable Demand Forecasting", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL is not None else "model not loaded",
        "rows_available": len(DATA) if DATA is not None else 0,
    }


@app.get("/items")
def list_items(limit: int = 20):
    """Sample of valid store-item combinations."""
    pairs = DATA[["store_nbr", "item_nbr", "family"]].drop_duplicates().head(limit)
    return {"count": len(pairs), "items": pairs.to_dict("records")}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    row = DATA[
        (DATA["store_nbr"] == req.store_nbr)
        & (DATA["item_nbr"] == req.item_nbr)
        & (DATA["date"] == pd.Timestamp(req.forecast_date))
    ]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for store {req.store_nbr}, item {req.item_nbr} "
                   f"on {req.forecast_date}. Try /items for valid combinations.",
        )

    pred = predict(MODEL, row, apply_cost_bias=req.apply_cost_bias)

    return ForecastResponse(
        store_nbr=req.store_nbr,
        item_nbr=req.item_nbr,
        forecast_date=req.forecast_date,
        family=str(row["family"].iloc[0]),
        predicted_units=round(float(pred[0]), 3),
        cost_bias_applied=req.apply_cost_bias,
    )
