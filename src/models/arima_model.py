"""
ARIMA forecasting baseline for selected forecasting datasets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100.0)


def evaluate_forecast(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }


def fit_arima_forecast(
    train_values: list[float],
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 0),
) -> np.ndarray:
    """
    Fit ARIMA on one series and forecast future values.

    Falls back to repeating the last value if ARIMA fails.
    """
    train_arr = np.asarray(train_values, dtype=float)

    if len(train_arr) < max(order) + 3:
        return np.full(horizon, train_arr[-1], dtype=float)

    try:
        model = ARIMA(train_arr, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)
    except Exception:
        return np.full(horizon, train_arr[-1], dtype=float)


def evaluate_arima_baseline(
    split_df: pd.DataFrame,
    order: tuple[int, int, int] = (1, 1, 0),
) -> pd.DataFrame:
    """
    Evaluate ARIMA on every series in a split dataframe.
    """
    rows: list[dict[str, Any]] = []

    for _, row in split_df.iterrows():
        series_id = row["series_id"]
        train_values = row["train_values"]
        test_values = row["test_values"]

        horizon = len(test_values)
        y_pred = fit_arima_forecast(train_values, horizon=horizon, order=order)
        metrics = evaluate_forecast(test_values, y_pred)

        rows.append({
            "series_id": series_id,
            "model": "arima",
            "order": str(order),
            "horizon": horizon,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "smape": metrics["smape"],
        })

    return pd.DataFrame(rows)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("model")[["mae", "rmse", "smape"]]
        .mean()
        .reset_index()
        .sort_values("mae")
        .reset_index(drop=True)
    )
    return summary