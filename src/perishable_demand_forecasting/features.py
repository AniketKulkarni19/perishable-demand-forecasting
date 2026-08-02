"""Feature engineering for perishable demand forecasting.

All historical features are computed on a series shifted by one day within
each store-item group, so no feature contains information from the target day.
"""

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
WINDOWS = [7, 14, 28]
GROUP_KEYS = ["store_nbr", "item_nbr"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. No leakage risk — dates are known in advance."""
    d = df["date"].dt
    df["dayofweek"] = d.dayofweek.astype("int8")
    df["day"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["year"] = d.year.astype("int16")
    df["weekofyear"] = d.isocalendar().week.astype("int8")
    df["is_weekend"] = d.dayofweek >= 5
    df["is_payday"] = (d.day == 15) | d.is_month_end
    df["days_in_month"] = d.days_in_month.astype("int8")
    return df


def add_lag_features(df: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """Lagged sales within each store-item pair."""
    grp = df.groupby(GROUP_KEYS, observed=True)["unit_sales"]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag).astype("float32")
    return df


def add_rolling_features(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """Rolling statistics computed on the shifted series to prevent leakage."""
    shifted = df.groupby(GROUP_KEYS, observed=True)["unit_sales"].shift(1)
    keys = [df[k] for k in GROUP_KEYS]
    for w in windows:
        roll = shifted.groupby(keys, observed=True).rolling(w, min_periods=1)
        for stat in ["mean", "std", "max"]:
            df[f"roll_{stat}_{w}"] = (
                getattr(roll, stat)()
                .reset_index(level=[0, 1], drop=True)
                .astype("float32")
            )
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Promotion history. Current-day promotion is known in advance."""
    grp = df.groupby(GROUP_KEYS, observed=True)["onpromotion"]
    df["promo_lag_1"] = grp.shift(1).astype("float32")
    shifted = grp.shift(1).astype("float32")
    keys = [df[k] for k in GROUP_KEYS]
    df["promo_count_28"] = (
        shifted.groupby(keys, observed=True)
        .rolling(28, min_periods=1).sum()
        .reset_index(level=[0, 1], drop=True)
        .astype("float32")
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline. Input must be sorted by store, item, date."""
    df = df.sort_values(GROUP_KEYS + ["date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_promo_features(df)
    return df