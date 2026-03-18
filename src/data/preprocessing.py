"""
Preprocessing utilities for selected forecasting datasets.

This module provides reusable functions to:
- validate dataset bundles
- prepare train/test splits
- optionally scale time series
- create supervised learning windows
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_dataset_bundle(bundle: dict[str, Any]) -> None:
    """
    Validate that a loaded dataset bundle contains the expected keys.
    """
    required_keys = {
        "dataset_key",
        "domain",
        "file_path",
        "metadata",
        "attribute_defs",
        "records_df",
        "long_df",
        "split_df",
    }

    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f"Dataset bundle is missing keys: {missing_keys}")

    if bundle["records_df"].empty:
        raise ValueError("records_df is empty")

    if bundle["split_df"].empty:
        raise ValueError("split_df is empty")


def fill_missing_values(values: list[float], method: str = "ffill") -> list[float]:
    """
    Fill missing values in a series.

    Parameters
    ----------
    values : list[float]
        Input series values.
    method : str
        Filling strategy: 'ffill', 'bfill', 'zero', or 'mean'.

    Returns
    -------
    list[float]
        Filled series values.
    """
    series = pd.Series(values, dtype="float64")

    if method == "ffill":
        series = series.ffill().bfill()
    elif method == "bfill":
        series = series.bfill().ffill()
    elif method == "zero":
        series = series.fillna(0.0)
    elif method == "mean":
        series = series.fillna(series.mean())
    else:
        raise ValueError(f"Unsupported missing value method: {method}")

    return series.tolist()


def scale_series_minmax(values: list[float]) -> tuple[list[float], dict[str, float]]:
    """
    Apply min-max scaling to a single series.

    Returns
    -------
    scaled_values : list[float]
    scaler_info : dict
        Contains min and max values for inverse scaling if needed.
    """
    arr = np.asarray(values, dtype=float)

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if max_val == min_val:
        scaled = np.zeros_like(arr, dtype=float)
    else:
        scaled = (arr - min_val) / (max_val - min_val)

    scaler_info = {"min": float(min_val), "max": float(max_val)}
    return scaled.tolist(), scaler_info


def scale_series_zscore(values: list[float]) -> tuple[list[float], dict[str, float]]:
    """
    Apply z-score scaling to a single series.

    Returns
    -------
    scaled_values : list[float]
    scaler_info : dict
        Contains mean and std for inverse scaling if needed.
    """
    arr = np.asarray(values, dtype=float)

    mean_val = np.nanmean(arr)
    std_val = np.nanstd(arr)

    if std_val == 0:
        scaled = np.zeros_like(arr, dtype=float)
    else:
        scaled = (arr - mean_val) / std_val

    scaler_info = {"mean": float(mean_val), "std": float(std_val)}
    return scaled.tolist(), scaler_info


def preprocess_split_dataframe(
    split_df: pd.DataFrame,
    fill_missing: bool = True,
    missing_method: str = "ffill",
    scaling: str | None = None,
) -> pd.DataFrame:
    """
    Preprocess the split dataframe produced by the dataset loader.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame with columns:
        - series_id
        - train_values
        - test_values
        - train_length
        - test_length
    fill_missing : bool
        Whether to fill missing values in train/test series.
    missing_method : str
        Strategy for missing value filling.
    scaling : str | None
        Optional scaling method: None, 'minmax', or 'zscore'.

    Returns
    -------
    pd.DataFrame
        Processed split dataframe.
    """
    processed_rows = []

    for _, row in split_df.iterrows():
        train_values = row["train_values"]
        test_values = row["test_values"]

        if fill_missing:
            train_values = fill_missing_values(train_values, method=missing_method)
            test_values = fill_missing_values(test_values, method=missing_method)

        scaler_info = None

        if scaling == "minmax":
            train_values, scaler_info = scale_series_minmax(train_values)
            # scale test using training min/max
            train_min = scaler_info["min"]
            train_max = scaler_info["max"]

            test_arr = np.asarray(test_values, dtype=float)
            if train_max == train_min:
                test_values = np.zeros_like(test_arr, dtype=float).tolist()
            else:
                test_values = ((test_arr - train_min) / (train_max - train_min)).tolist()

        elif scaling == "zscore":
            train_values, scaler_info = scale_series_zscore(train_values)
            # scale test using training mean/std
            train_mean = scaler_info["mean"]
            train_std = scaler_info["std"]

            test_arr = np.asarray(test_values, dtype=float)
            if train_std == 0:
                test_values = np.zeros_like(test_arr, dtype=float).tolist()
            else:
                test_values = ((test_arr - train_mean) / train_std).tolist()

        elif scaling is not None:
            raise ValueError(f"Unsupported scaling option: {scaling}")

        processed_rows.append({
            "series_id": row["series_id"],
            "train_values": train_values,
            "test_values": test_values,
            "train_length": len(train_values),
            "test_length": len(test_values),
            "scaler_info": scaler_info,
        })

    return pd.DataFrame(processed_rows)


def create_supervised_windows(
    series_values: list[float],
    input_window: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one time series into supervised learning windows.

    Parameters
    ----------
    series_values : list[float]
        Input series.
    input_window : int
        Number of past time steps used as input.
    forecast_horizon : int
        Number of future time steps to predict.

    Returns
    -------
    X : np.ndarray
        Input windows.
    y : np.ndarray
        Forecast targets.
    """
    values = np.asarray(series_values, dtype=float)

    X, y = [], []

    max_start = len(values) - input_window - forecast_horizon + 1
    for start_idx in range(max_start):
        end_input = start_idx + input_window
        end_target = end_input + forecast_horizon

        X.append(values[start_idx:end_input])
        y.append(values[end_input:end_target])

    if not X:
        return np.empty((0, input_window)), np.empty((0, forecast_horizon))

    return np.asarray(X), np.asarray(y)


def build_supervised_dataset(
    split_df: pd.DataFrame,
    input_window: int,
    use_train_only: bool = True,
) -> dict[str, np.ndarray]:
    """
    Build a supervised learning dataset from a split dataframe.

    Parameters
    ----------
    split_df : pd.DataFrame
        Preprocessed split dataframe.
    input_window : int
        Number of past steps used as input.
    use_train_only : bool
        If True, build windows from train_values only.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing concatenated X and y arrays.
    """
    all_X = []
    all_y = []

    for _, row in split_df.iterrows():
        train_values = row["train_values"]
        test_values = row["test_values"]

        if use_train_only:
            series_values = train_values
            forecast_horizon = len(test_values)
        else:
            series_values = train_values + test_values
            forecast_horizon = len(test_values)

        X, y = create_supervised_windows(
            series_values=series_values,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
        )

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return {
            "X": np.empty((0, input_window)),
            "y": np.empty((0, 0)),
        }

    return {
        "X": np.vstack(all_X),
        "y": np.vstack(all_y),
    }