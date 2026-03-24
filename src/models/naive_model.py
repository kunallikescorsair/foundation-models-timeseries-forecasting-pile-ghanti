"""
Naive forecasting baselines for time series forecasting.

This module provides:
- naive forecast
- seasonal naive forecast
- seasonal period inference
- batch evaluation on split dataframes
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_forecast
from src.models.base_model import BaseForecastModel

def naive_forecast(train_values: list[float], horizon: int) -> np.ndarray:
    """
    Forecast by repeating the last observed value.

    Parameters
    ----------
    train_values : list[float]
        Historical training values.
    horizon : int
        Number of future steps to forecast.

    Returns
    -------
    np.ndarray
        Forecast array of length `horizon`.
    """
    if len(train_values) == 0:
        raise ValueError("train_values must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    last_value = float(train_values[-1])
    return np.full(shape=horizon, fill_value=last_value, dtype=float)


def seasonal_naive_forecast(
    train_values: list[float],
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """
    Forecast by repeating the last seasonal cycle.

    Parameters
    ----------
    train_values : list[float]
        Historical training values.
    horizon : int
        Number of future steps to forecast.
    seasonal_period : int
        Seasonal length.

    Returns
    -------
    np.ndarray
        Forecast array of length `horizon`.
    """
    if len(train_values) == 0:
        raise ValueError("train_values must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")

    train_arr = np.asarray(train_values, dtype=float)

    # Fall back to the naive forecast if history is too short.
    if len(train_arr) < seasonal_period:
        return naive_forecast(train_values, horizon)

    last_season = train_arr[-seasonal_period:]
    repeats = int(np.ceil(horizon / seasonal_period))
    forecast = np.tile(last_season, repeats)[:horizon]

    return forecast.astype(float)


def infer_seasonal_period(
    dataset_key: str,
    frequency: str | None = None,
) -> int:
    """
    Infer a practical seasonal period for a baseline seasonal-naive forecast.

    Parameters
    ----------
    dataset_key : str
        Normalized dataset key.
    frequency : str | None
        Normalized frequency label if available.

    Returns
    -------
    int
        Seasonal period.
    """
    normalized_frequency = None if frequency is None else frequency.strip().lower()

    frequency_map = {
        "yearly": 1,
        "quarterly": 4,
        "monthly": 12,
        "weekly": 52,
        "daily": 7,
        "hourly": 24,
        "half_hourly": 48,
        "10_minutes": 144,
        "minutely": 1440,
        "4_seconds": 1,
        "seconds": 1,
    }

    if normalized_frequency in frequency_map:
        return frequency_map[normalized_frequency]

    key = dataset_key.lower()

    if "yearly" in key:
        return 1
    if "quarterly" in key:
        return 4
    if "monthly" in key:
        return 12
    if "weekly" in key:
        return 52
    if "daily" in key:
        return 7
    if "hourly" in key:
        return 24
    if "half_hourly" in key:
        return 48
    if "10_minutes" in key:
        return 144

    return 1


def evaluate_naive_baseline(
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the naive forecast on every series in a split dataframe.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame with:
        - series_id
        - train_values
        - test_values

    Returns
    -------
    pd.DataFrame
        Per-series results.
    """
    rows: list[dict[str, Any]] = []

    for _, row in split_df.iterrows():
        series_id = row["series_id"]
        train_values = row["train_values"]
        test_values = row["test_values"]

        horizon = len(test_values)
        y_pred = naive_forecast(train_values, horizon)
        metrics = evaluate_forecast(test_values, y_pred)

        rows.append(
            {
                "series_id": series_id,
                "model": "naive",
                "horizon": horizon,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "smape": metrics["smape"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_seasonal_naive_baseline(
    split_df: pd.DataFrame,
    seasonal_period: int,
) -> pd.DataFrame:
    """
    Evaluate the seasonal naive forecast on every series in a split dataframe.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame with:
        - series_id
        - train_values
        - test_values
    seasonal_period : int
        Seasonal period used for the forecast.

    Returns
    -------
    pd.DataFrame
        Per-series results.
    """
    rows: list[dict[str, Any]] = []

    for _, row in split_df.iterrows():
        series_id = row["series_id"]
        train_values = row["train_values"]
        test_values = row["test_values"]

        horizon = len(test_values)
        used_naive_fallback = len(train_values) < seasonal_period

        y_pred = seasonal_naive_forecast(train_values, horizon, seasonal_period)
        metrics = evaluate_forecast(test_values, y_pred)

        rows.append(
            {
                "series_id": series_id,
                "model": "seasonal_naive",
                "seasonal_period": seasonal_period,
                "used_naive_fallback": used_naive_fallback,
                "horizon": horizon,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "smape": metrics["smape"],
            }
        )

    return pd.DataFrame(rows)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-series results into summary metrics.
    """
    summary = (
        results_df.groupby("model", as_index=False)
        .agg(
            n_series=("series_id", "count"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
        )
        .sort_values("mae")
        .reset_index(drop=True)
    )
    return summary

class NaiveModel(BaseForecastModel):
    """
    Last-value naive forecast.
    """

    name = "naive"

    def __init__(self) -> None:
        self.train_values: list[float] | None = None

    def fit(self, train_values: list[float]) -> None:
        if len(train_values) == 0:
            raise ValueError("train_values must not be empty")
        self.train_values = train_values

    def predict(self, horizon: int) -> list[float]:
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")
        return naive_forecast(self.train_values, horizon).tolist()

class SeasonalNaiveModel(BaseForecastModel):
    """
    Seasonal naive forecast.
    """

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int) -> None:
        self.seasonal_period = seasonal_period
        self.train_values: list[float] | None = None

    def fit(self, train_values: list[float]) -> None:
        if len(train_values) == 0:
            raise ValueError("train_values must not be empty")
        self.train_values = train_values

    def predict(self, horizon: int) -> list[float]:
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")
        return seasonal_naive_forecast(
            self.train_values,
            horizon,
            self.seasonal_period,
        ).tolist()

    def get_params(self) -> dict[str, Any]:
        return {"seasonal_period": self.seasonal_period}