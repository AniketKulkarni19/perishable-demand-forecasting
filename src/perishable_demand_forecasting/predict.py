"""Model loading and prediction for perishable demand forecasting."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lgbm_holidays.txt"

CATEGORICALS = [
    "family", "city", "state", "store_type",
    "store_nbr", "item_nbr", "class", "cluster",
]

# Selected on validation data, applied unchanged to test.
# Minimizes asymmetric spoilage cost rather than forecast error.
COST_OPTIMAL_BIAS = 0.80


def load_model(path: Path = MODEL_PATH) -> lgb.Booster:
    """Load the trained LightGBM booster."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return lgb.Booster(model_file=str(path))


def prepare_input(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Cast categoricals and align columns to the model's expected order."""
    df = df.copy()
    for col in CATEGORICALS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[feature_names]


def predict(
    model: lgb.Booster,
    df: pd.DataFrame,
    apply_cost_bias: bool = False,
) -> np.ndarray:
    """Predict unit sales.

    The model is trained on a log1p-transformed target, so predictions are
    inverted with expm1 and floored at zero.

    Set apply_cost_bias=True to scale by the cost-optimal factor, which
    reduces spoilage cost at the expense of forecast accuracy.
    """
    X = prepare_input(df, model.feature_name())
    preds = np.clip(np.expm1(model.predict(X)), 0, None)

    if apply_cost_bias:
        preds = preds * COST_OPTIMAL_BIAS

    return preds
