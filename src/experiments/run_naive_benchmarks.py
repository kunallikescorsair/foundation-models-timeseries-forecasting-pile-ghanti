"""
Run naive and seasonal naive benchmarks across all selected datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset_loader import (
    build_dataset_summary,
    list_available_datasets,
    load_monash_dataset,
)
from src.data.preprocessing import preprocess_split_dataframe
from src.models.naive_model import (
    evaluate_naive_baseline,
    evaluate_seasonal_naive_baseline,
    infer_seasonal_period,
    summarize_results,
)


def run_naive_benchmarks(
    output_dir: str | Path = "results/tables/naive_benchmarks",
    fill_missing: bool = True,
    missing_method: str = "ffill",
    scaling: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run naive and seasonal naive benchmarks over all selected datasets.

    Returns
    -------
    per_dataset_summary_df : pd.DataFrame
        Aggregated summary metrics by dataset and model.
    overall_summary_df : pd.DataFrame
        Aggregated summary metrics by model across all datasets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys = list_available_datasets()

    all_per_series_results = []
    per_dataset_summary_rows = []

    print("Running naive benchmarks on selected datasets...")
    print(dataset_keys)

    for dataset_key in dataset_keys:
        print(f"\nProcessing dataset: {dataset_key}")

        bundle = load_monash_dataset(dataset_key)

        processed_split_df = preprocess_split_dataframe(
            bundle["split_df"],
            fill_missing=fill_missing,
            missing_method=missing_method,
            scaling=scaling,
        )

        # Naive baseline
        naive_results_df = evaluate_naive_baseline(processed_split_df)
        naive_results_df["dataset_key"] = dataset_key
        naive_results_df["domain"] = bundle["domain"]
        naive_results_df["frequency"] = bundle["metadata"]["frequency"]

        # Seasonal naive baseline
        seasonal_period = infer_seasonal_period(
            dataset_key=bundle["dataset_key"],
            frequency=bundle["metadata"]["frequency"],
        )

        seasonal_results_df = evaluate_seasonal_naive_baseline(
            processed_split_df,
            seasonal_period=seasonal_period,
        )
        seasonal_results_df["dataset_key"] = dataset_key
        seasonal_results_df["domain"] = bundle["domain"]
        seasonal_results_df["frequency"] = bundle["metadata"]["frequency"]

        # Save per-series results
        naive_results_df.to_csv(
            output_dir / f"{dataset_key}_naive_results.csv",
            index=False,
        )
        seasonal_results_df.to_csv(
            output_dir / f"{dataset_key}_seasonal_naive_results.csv",
            index=False,
        )

        dataset_results_df = pd.concat(
            [naive_results_df, seasonal_results_df],
            ignore_index=True,
        )

        # Per-dataset summary
        dataset_summary_df = (
            dataset_results_df.groupby(
                ["dataset_key", "domain", "frequency", "model"], as_index=False
            )[["mae", "rmse", "smape"]]
            .mean()
            .sort_values(["dataset_key", "mae"])
            .reset_index(drop=True)
        )

        dataset_summary_df.to_csv(
            output_dir / f"{dataset_key}_summary.csv",
            index=False,
        )

        per_dataset_summary_rows.append(dataset_summary_df)
        all_per_series_results.append(dataset_results_df)

    all_per_series_results_df = pd.concat(all_per_series_results, ignore_index=True)
    per_dataset_summary_df = pd.concat(per_dataset_summary_rows, ignore_index=True)

    # Overall summary across all datasets
    overall_summary_df = (
        all_per_series_results_df.groupby("model", as_index=False)[["mae", "rmse", "smape"]]
        .mean()
        .sort_values("mae")
        .reset_index(drop=True)
    )

    # Save combined outputs
    all_per_series_results_df.to_csv(output_dir / "all_per_series_results.csv", index=False)
    per_dataset_summary_df.to_csv(output_dir / "per_dataset_summary.csv", index=False)
    overall_summary_df.to_csv(output_dir / "overall_summary.csv", index=False)

    # Also save dataset metadata summary
    dataset_info_df = build_dataset_summary()
    dataset_info_df.to_csv(output_dir / "dataset_info_summary.csv", index=False)

    print("\nBenchmarking complete.")
    print(f"Saved outputs to: {output_dir}")

    return per_dataset_summary_df, overall_summary_df


if __name__ == "__main__":
    per_dataset_summary_df, overall_summary_df = run_naive_benchmarks()

    print("\nPer-dataset summary:")
    print(per_dataset_summary_df.head(20))

    print("\nOverall summary:")
    print(overall_summary_df)