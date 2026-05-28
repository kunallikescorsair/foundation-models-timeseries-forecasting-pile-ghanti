"""
Shared benchmarking utilities for forecasting models.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.evaluation.metrics import evaluate_forecast
from src.models.base_model import BaseForecastModel


def evaluate_model_on_split_dataframe(
    model: BaseForecastModel,
    split_df: pd.DataFrame,
    dataset_key: str,
    domain: str,
    frequency: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Evaluate one forecasting model across all series in a split dataframe.

    Records:
    - fit_time_seconds
    - inference_time_seconds
    - series_total_time_seconds
    """
    rows: list[dict[str, Any]] = []
    extra_metadata = extra_metadata or {}

    for _, row in split_df.iterrows():
        series_id = row["series_id"]
        train_values = row["train_values"]
        test_values = row["test_values"]

        horizon = len(test_values)

        fit_time_seconds = None
        inference_time_seconds = None
        series_total_time_seconds = None

        try:
            series_start_time = time.time()

            fit_start_time = time.time()
            model.fit(train_values)
            fit_time_seconds = time.time() - fit_start_time

            inference_start_time = time.time()
            y_pred = model.predict(horizon)
            inference_time_seconds = time.time() - inference_start_time

            series_total_time_seconds = time.time() - series_start_time

            metrics = evaluate_forecast(test_values, y_pred)

            result_row = {
                "series_id": series_id,
                "dataset_key": dataset_key,
                "domain": domain,
                "frequency": frequency,
                "model": model.name,
                "horizon": horizon,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "smape": metrics["smape"],
                "fit_time_seconds": fit_time_seconds,
                "inference_time_seconds": inference_time_seconds,
                "series_total_time_seconds": series_total_time_seconds,
                "status": "success",
                "error_message": None,
            }
            result_row.update(extra_metadata)
            result_row.update(model.get_params())

        except Exception as exc:
            result_row = {
                "series_id": series_id,
                "dataset_key": dataset_key,
                "domain": domain,
                "frequency": frequency,
                "model": model.name,
                "horizon": horizon,
                "mae": None,
                "rmse": None,
                "smape": None,
                "fit_time_seconds": fit_time_seconds,
                "inference_time_seconds": inference_time_seconds,
                "series_total_time_seconds": series_total_time_seconds,
                "status": "failed",
                "error_message": str(exc),
            }
            result_row.update(extra_metadata)
            result_row.update(model.get_params())

        rows.append(result_row)

    return pd.DataFrame(rows)


def summarize_benchmark_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate successful per-series results into summary metrics.
    """
    successful_df = results_df[results_df["status"] == "success"].copy()

    if successful_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "domain",
                "frequency",
                "model",
                "n_series",
                "mae",
                "rmse",
                "smape",
                "fit_time_seconds",
                "inference_time_seconds",
                "series_total_time_seconds",
            ]
        )

    agg_dict = {
        "n_series": ("series_id", "count"),
        "mae": ("mae", "mean"),
        "rmse": ("rmse", "mean"),
        "smape": ("smape", "mean"),
    }

    if "fit_time_seconds" in successful_df.columns:
        agg_dict["fit_time_seconds"] = ("fit_time_seconds", "mean")

    if "inference_time_seconds" in successful_df.columns:
        agg_dict["inference_time_seconds"] = ("inference_time_seconds", "mean")

    if "series_total_time_seconds" in successful_df.columns:
        agg_dict["series_total_time_seconds"] = ("series_total_time_seconds", "mean")

    summary_df = (
        successful_df.groupby(
            ["dataset_key", "domain", "frequency", "model"],
            as_index=False,
        )
        .agg(**agg_dict)
        .sort_values(["dataset_key", "smape"])
        .reset_index(drop=True)
    )

    return summary_df