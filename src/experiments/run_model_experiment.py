"""
Run a single forecasting experiment.

Experiment unit:
dataset_key + model_name + horizon + horizon_type + experiment_type

This runner records:
- accuracy metrics
- fit time
- inference time
- total runtime
- peak memory usage
- model parameters
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_loader import load_monash_dataset
from src.data.preprocessing import preprocess_split_dataframe
from src.evaluation.benchmarking import evaluate_model_on_split_dataframe
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.naive_model import NaiveModel, SeasonalNaiveModel
from src.models.prophet_model import ProphetModel
from src.models.timesfm_model import TimesFMModel
from src.models.ttm_model import TTMModel
from src.models.uni2ts_model import Uni2TSModel

def get_default_seasonal_period(freq: str | None) -> int:
    if freq is None:
        return 1

    freq = freq.lower()

    mapping = {
        "10_minutes": 144,
        "half_hourly": 48,
        "hourly": 24,
        "daily": 7,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }

    return mapping.get(freq, 1)

def build_model(
    model_name: str,
    normalized_frequency: str,
    model_params: dict[str, Any] | None = None,
):
    """
    Build a model object from model name and optional parameters.
    """
    model_params = model_params or {}
    model_name = model_name.lower().strip()

    if model_name == "naive":
        return NaiveModel(**model_params)

    if model_name == "seasonal_naive":
        if "seasonal_period" not in model_params:
            model_params["seasonal_period"] = get_default_seasonal_period(normalized_frequency)

        return SeasonalNaiveModel(**model_params)

    if model_name == "arima":
        return ARIMAModel(**model_params)

    if model_name == "prophet":
        return ProphetModel(**model_params)

    if model_name == "lstm":
        return LSTMModel(**model_params)

    if model_name == "timesfm":
        return TimesFMModel(**model_params)

    if model_name == "ttm":
        return TTMModel(
            normalized_frequency=normalized_frequency,
            **model_params,
        )

    if model_name == "uni2ts":
        return Uni2TSModel(
            normalized_frequency=normalized_frequency,
            **model_params,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def override_split_horizon(split_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Override train/test split using a new forecast horizon.

    Assumes split_df contains:
    - series_id
    - values or series_values OR train_values/test_values depending on pipeline

    This function supports the current project split format where each row
    contains train_values and test_values after loading/preprocessing.
    """
    updated_rows = []

    for _, row in split_df.iterrows():
        full_values = None

        if "series_values" in row:
            full_values = row["series_values"]
        elif "values" in row:
            full_values = row["values"]
        elif "train_values" in row and "test_values" in row:
            full_values = list(row["train_values"]) + list(row["test_values"])
        else:
            raise ValueError(
                "Could not reconstruct full series. Expected one of: "
                "'series_values', 'values', or train_values + test_values."
            )

        full_values = list(full_values)

        if len(full_values) <= horizon:
            raise ValueError(
                f"Series {row.get('series_id', 'unknown')} is too short for horizon={horizon}"
            )

        new_row = row.copy()
        new_row["train_values"] = full_values[:-horizon]
        new_row["test_values"] = full_values[-horizon:]
        new_row["horizon"] = horizon

        updated_rows.append(new_row)

    return pd.DataFrame(updated_rows)


def run_model_experiment(
    dataset_key: str,
    model_name: str,
    horizon: int | None = None,
    horizon_type: str = "default",
    experiment_type: str = "zero_shot",
    output_dir: str | Path = "results/raw_runs",
    fill_missing: bool = True,
    missing_method: str = "ffill",
    scaling: str | None = None,
    series_limit: int | None = None,
    model_params: dict[str, Any] | None = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run one model on one dataset for one horizon setting.

    Parameters
    ----------
    dataset_key : str
        Dataset key from project configuration.
    model_name : str
        Model name, e.g. 'arima', 'timesfm', 'lstm'.
    horizon : int | None
        Forecast horizon. If None, uses dataset resolved horizon.
    horizon_type : str
        Label such as 'short', 'long', or 'default'.
    experiment_type : str
        Label such as 'baseline', 'zero_shot', 'tuned', or 'finetuned'.
    output_dir : str | Path
        Base output directory.
    fill_missing : bool
        Whether to fill missing values.
    missing_method : str
        Missing value method.
    scaling : str | None
        Optional scaling method.
    series_limit : int | None
        Optional series subset for debugging.
    model_params : dict[str, Any] | None
        Parameters passed to model constructor.
    save_results : bool
        Whether to save CSV output.

    Returns
    -------
    pd.DataFrame
        Per-series experiment results.
    """
    model_params = model_params or {}

    output_dir = Path(output_dir)

    bundle = load_monash_dataset(dataset_key)

    processed_split_df = preprocess_split_dataframe(
        bundle["split_df"],
        fill_missing=fill_missing,
        missing_method=missing_method,
        scaling=scaling,
    )

    if horizon is None:
        horizon = int(bundle["resolved_horizon"])

    if horizon != int(bundle["resolved_horizon"]):
        processed_split_df = override_split_horizon(processed_split_df, horizon)

    if series_limit is not None:
        processed_split_df = processed_split_df.head(series_limit).copy()

    if processed_split_df.empty:
        raise ValueError("Processed split dataframe is empty")

    model = build_model(
        model_name=model_name,
        normalized_frequency=bundle["normalized_frequency"],
        model_params=model_params,
    )

    tracemalloc.start()
    total_start = time.time()

    results_df = evaluate_model_on_split_dataframe(
        model=model,
        split_df=processed_split_df,
        dataset_key=bundle["dataset_key"],
        domain=bundle["domain"],
        frequency=bundle["normalized_frequency"],
        extra_metadata={
            "resolved_horizon": horizon,
            "original_resolved_horizon": bundle["resolved_horizon"],
            "horizon_strategy": bundle["horizon_strategy"],
            "horizon_type": horizon_type,
            "experiment_type": experiment_type,
            "model_params_json": json.dumps(model_params, default=str),
        },
    )

    total_runtime_seconds = time.time() - total_start
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_memory / (1024 * 1024)

    results_df["total_runtime_seconds"] = total_runtime_seconds
    results_df["runtime_per_series_seconds"] = (
        total_runtime_seconds / len(processed_split_df)
        if len(processed_split_df) > 0
        else None
    )
    results_df["peak_memory_mb"] = peak_memory_mb
    results_df["n_series_evaluated"] = len(processed_split_df)
    results_df["horizon_type"] = horizon_type
    results_df["experiment_type"] = experiment_type
    results_df["model_params_json"] = json.dumps(model_params, default=str)

    if save_results:
        save_dir = output_dir / bundle["domain"] / dataset_key
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_model_name = model_name.lower().strip()
        filename = f"{safe_model_name}_{horizon_type}_h{horizon}_{experiment_type}.csv"

        results_df.to_csv(save_dir / filename, index=False)

        print(f"Saved: {save_dir / filename}")

    return results_df


if __name__ == "__main__":
    df = run_model_experiment(
        dataset_key="electricity_weekly",
        model_name="timesfm",
        horizon=8,
        horizon_type="long",
        experiment_type="zero_shot",
        series_limit=3,
    )

    print(df.head())