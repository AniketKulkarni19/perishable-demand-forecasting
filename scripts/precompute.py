import sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from perishable_demand_forecasting.predict import load_model, predict, COST_OPTIMAL_BIAS

data = pd.read_parquet(ROOT / "data" / "serving" / "features.parquet")
data["date"] = pd.to_datetime(data["date"])
model = load_model()

print(f"Predicting {len(data):,} rows...")
data["forecast"] = predict(model, data)
data["forecast_cost_optimal"] = data["forecast"] * COST_OPTIMAL_BIAS

out = data[[
    "date", "store_nbr", "item_nbr", "family", "city", "store_type",
    "unit_sales", "onpromotion", "forecast", "forecast_cost_optimal",
]].copy()

out.to_parquet(ROOT / "data" / "serving" / "predictions.parquet", index=False)
print(f"Saved {len(out):,} predictions")
print(out.head())
