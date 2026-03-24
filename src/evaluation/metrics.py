"""
Shared evaluation metrics for time series forecasting.

This module provides reusable metric functions for:
- MAE
- RMSE
- sMAPE
- forecast input validation
"""

from __future__ import annotations

import numpy as np


def validate_forecast_inputs(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    allow_nan: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert forecast inputs to numpy arrays.

    Parameters
    ----------
    y_true : list[float] | np.ndarray
        Ground-truth target values.
    y_pred : list[float] | np.ndarray
        Predicted values.
    allow_nan : bool
        Whether NaN values are allowed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated y_true and y_pred arrays.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape}, y_pred {y_pred_arr.shape}"
        )

    if y_true_arr.ndim != 1:
        y_true_arr = y_true_arr.reshape(-1)
        y_pred_arr = y_pred_arr.reshape(-1)

    if len(y_true_arr) == 0:
        raise ValueError("Forecast arrays must not be empty")

    if not allow_nan:
        if not np.all(np.isfinite(y_true_arr)):
            raise ValueError("y_true contains non-finite values")
        if not np.all(np.isfinite(y_pred_arr)):
            raise ValueError("y_pred contains non-finite values")

    return y_true_arr, y_pred_arr


def mean_absolute_error(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> float:
    """
    Compute mean absolute error.
    """
    y_true_arr, y_pred_arr = validate_forecast_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def root_mean_squared_error(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> float:
    """
    Compute root mean squared error.
    """
    y_true_arr, y_pred_arr = validate_forecast_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def symmetric_mean_absolute_percentage_error(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute symmetric mean absolute percentage error in percentage form.
    """
    y_true_arr, y_pred_arr = validate_forecast_inputs(y_true, y_pred)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr) + epsilon
    return float(np.mean(2.0 * np.abs(y_pred_arr - y_true_arr) / denominator) * 100.0)


def evaluate_forecast(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> dict[str, float]:
    """
    Evaluate one forecast using standard metrics.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }