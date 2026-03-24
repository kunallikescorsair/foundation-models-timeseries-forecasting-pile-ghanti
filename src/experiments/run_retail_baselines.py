"""
Run baseline models on the selected retail datasets.

This script currently runs:
- Naive
- Seasonal Naive

It is designed to provide a consistent retail-domain benchmark runner.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset_loader import load_monash_dataset
from src.data.preprocessing import preprocess_split_dataframe
from src.data.project_config import RETAIL_DATASETS
from src.evaluation.benchmarking import (
    evaluate_model_on_split_dataframe,
    summarize_benchmark_results,
)
from src.models.naive_model import (
    NaiveModel,
    SeasonalNaiveModel,
    infer_seasonal_period,
)


def run_retail_baselines(
    output_dir: str | Path = "results/tables/retail_baselines",
    fill_missing: bool = True,
    missing_method: str = "ffill",
    scaling: str | None = None,
    series_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run baseline models on selected retail datasets.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save outputs.
    fill_missing : bool
        Whether to fill missing values.
    missing_method : str
        Missing value filling method.
    scaling : str | None
        Optional scaling method.
    series_limit : int | None
        Optional limit on number of series per dataset for quick testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Per-series results, summary results, run status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []
    run_status_rows = []

    for dataset_key in RETAIL_DATASETS:
        print(f"\nProcessing retail dataset: {dataset_key}")

        try:
            bundle = load_monash_dataset(dataset_key)

            processed_split_df = preprocess_split_dataframe(
                bundle["split_df"],
                fill_missing=fill_missing,
                missing_method=missing_method,
                scaling=scaling,
            )

            if series_limit is not None:
                processed_split_df = processed_split_df.head(series_limit).copy()

            if processed_split_df.empty:
                raise ValueError("Processed split dataframe is empty")

            seasonal_period = infer_seasonal_period(
                dataset_key=bundle["dataset_key"],
                frequency=bundle["normalized_frequency"],
            )

            models = [
                NaiveModel(),
                SeasonalNaiveModel(seasonal_period=seasonal_period),
            ]

            model_results = []

            for model in models:
                print(f"  Running model: {model.name}")

                results_df = evaluate_model_on_split_dataframe(
                    model=model,
                    split_df=processed_split_df,
                    dataset_key=bundle["dataset_key"],
                    domain=bundle["domain"],
                    frequency=bundle["normalized_frequency"],
                    extra_metadata={
                        "resolved_horizon": bundle["resolved_horizon"],
                        "horizon_strategy": bundle["horizon_strategy"],
                    },
                )

                results_df.to_csv(
                    output_dir / f"{dataset_key}_{model.name}_results.csv",
                    index=False,
                )

                model_results.append(results_df)

            dataset_results_df = pd.concat(model_results, ignore_index=True)
            dataset_summary_df = summarize_benchmark_results(dataset_results_df)
            dataset_summary_df["resolved_horizon"] = bundle["resolved_horizon"]
            dataset_summary_df["horizon_strategy"] = bundle["horizon_strategy"]

            dataset_summary_df.to_csv(
                output_dir / f"{dataset_key}_summary.csv",
                index=False,
            )

            all_results.append(dataset_results_df)
            summary_rows.append(dataset_summary_df)

            run_status_rows.append(
                {
                    "dataset_key": dataset_key,
                    "status": "success",
                    "domain": bundle["domain"],
                    "frequency": bundle["normalized_frequency"],
                    "resolved_horizon": bundle["resolved_horizon"],
                    "horizon_strategy": bundle["horizon_strategy"],
                    "num_series_evaluated": len(processed_split_df),
                    "error_message": None,
                }
            )

        except Exception as exc:
            print(f"Failed dataset: {dataset_key} -> {exc}")

            run_status_rows.append(
                {
                    "dataset_key": dataset_key,
                    "status": "failed",
                    "domain": "retail",
                    "frequency": None,
                    "resolved_horizon": None,
                    "horizon_strategy": None,
                    "num_series_evaluated": None,
                    "error_message": str(exc),
                }
            )

    run_status_df = pd.DataFrame(run_status_rows)
    run_status_df.to_csv(output_dir / "run_status.csv", index=False)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        summary_df = pd.concat(summary_rows, ignore_index=True)

        overall_summary_df = (
            all_results_df[all_results_df["status"] == "success"]
            .groupby("model", as_index=False)[["mae", "rmse", "smape"]]
            .mean()
            .sort_values("mae")
            .reset_index(drop=True)
        )

        all_results_df.to_csv(output_dir / "all_per_series_results.csv", index=False)
        summary_df.to_csv(output_dir / "per_dataset_summary.csv", index=False)
        overall_summary_df.to_csv(output_dir / "overall_summary.csv", index=False)
    else:
        all_results_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        overall_summary_df = pd.DataFrame()

    print("\nRetail baseline run complete.")
    print(f"Saved outputs to: {output_dir}")

    return all_results_df, summary_df, run_status_df


if __name__ == "__main__":
    all_results_df, summary_df, run_status_df = run_retail_baselines(series_limit=20)

    print("\nRun status:")
    print(run_status_df)

    print("\nSummary:")
    print(summary_df.head())