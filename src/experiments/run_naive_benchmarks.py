"""
Run naive and seasonal naive benchmarks across Monash datasets.

This script:
- discovers datasets automatically
- loads each dataset safely
- preprocesses split data
- evaluates naive and seasonal naive baselines
- saves per-series and summary outputs
- logs failures without stopping the full run
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset_loader import (
    DEFAULT_MONASH_BASE_DIR,
    build_dataset_summary,
    list_available_datasets,
    load_monash_dataset,
)
from src.data.preprocessing import preprocess_split_dataframe
from src.models.naive_model import (
    evaluate_naive_baseline,
    evaluate_seasonal_naive_baseline,
    infer_seasonal_period,
)


def run_naive_benchmarks(
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR,
    output_dir: str | Path = "results/tables/naive_benchmarks",
    include_datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
    fill_missing: bool = True,
    missing_method: str = "ffill",
    scaling: str | None = None,
    fallback_ratio: float = 0.1,
    min_horizon: int = 1,
    max_horizon: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run naive and seasonal naive benchmarks across Monash datasets.

    Parameters
    ----------
    base_dir : str | Path
        Monash dataset directory.
    output_dir : str | Path
        Output directory for benchmark results.
    include_datasets : list[str] | None
        Optional dataset whitelist.
    exclude_datasets : list[str] | None
        Optional dataset blacklist.
    fill_missing : bool
        Whether to fill missing values.
    missing_method : str
        Missing value filling method.
    scaling : str | None
        Optional scaling method.
    fallback_ratio : float
        Ratio used when metadata horizon is missing.
    min_horizon : int
        Minimum fallback horizon.
    max_horizon : int | None
        Optional cap for fallback horizon.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Per-dataset summary, overall summary, and run status dataframe.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys = list_available_datasets(base_dir)

    if include_datasets is not None:
        include_set = set(include_datasets)
        dataset_keys = [key for key in dataset_keys if key in include_set]

    if exclude_datasets is not None:
        exclude_set = set(exclude_datasets)
        dataset_keys = [key for key in dataset_keys if key not in exclude_set]

    print("Running naive benchmarks on Monash datasets...")
    print(dataset_keys)

    all_per_series_results: list[pd.DataFrame] = []
    per_dataset_summary_rows: list[pd.DataFrame] = []
    run_status_rows: list[dict[str, object]] = []

    for dataset_key in dataset_keys:
        print(f"\nProcessing dataset: {dataset_key}")

        try:
            bundle = load_monash_dataset(
                dataset_key=dataset_key,
                base_dir=base_dir,
                fallback_ratio=fallback_ratio,
                min_horizon=min_horizon,
                max_horizon=max_horizon,
            )

            processed_split_df = preprocess_split_dataframe(
                bundle["split_df"],
                fill_missing=fill_missing,
                missing_method=missing_method,
                scaling=scaling,
            )

            if processed_split_df.empty:
                raise ValueError("Processed split dataframe is empty")

            frequency = bundle["normalized_frequency"]
            seasonal_period = infer_seasonal_period(
                dataset_key=bundle["dataset_key"],
                frequency=frequency,
            )

            # Run baselines.
            naive_results_df = evaluate_naive_baseline(processed_split_df)
            seasonal_results_df = evaluate_seasonal_naive_baseline(
                processed_split_df,
                seasonal_period=seasonal_period,
            )

            # Add dataset-level metadata.
            for df in (naive_results_df, seasonal_results_df):
                df["dataset_key"] = dataset_key
                df["domain"] = bundle["domain"]
                df["frequency"] = frequency
                df["resolved_horizon"] = bundle["resolved_horizon"]
                df["horizon_strategy"] = bundle["horizon_strategy"]

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

            dataset_summary_df = (
                dataset_results_df.groupby(
                    [
                        "dataset_key",
                        "domain",
                        "frequency",
                        "resolved_horizon",
                        "horizon_strategy",
                        "model",
                    ],
                    as_index=False,
                )
                .agg(
                    n_series=("series_id", "count"),
                    mae=("mae", "mean"),
                    rmse=("rmse", "mean"),
                    smape=("smape", "mean"),
                )
                .sort_values(["dataset_key", "mae"])
                .reset_index(drop=True)
            )

            dataset_summary_df.to_csv(
                output_dir / f"{dataset_key}_summary.csv",
                index=False,
            )

            per_dataset_summary_rows.append(dataset_summary_df)
            all_per_series_results.append(dataset_results_df)

            run_status_rows.append(
                {
                    "dataset_key": dataset_key,
                    "status": "success",
                    "file_name": bundle["file_path"].name,
                    "domain": bundle["domain"],
                    "frequency": frequency,
                    "resolved_horizon": bundle["resolved_horizon"],
                    "horizon_strategy": bundle["horizon_strategy"],
                    "num_series_loaded": len(bundle["records_df"]),
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
                    "file_name": None,
                    "domain": None,
                    "frequency": None,
                    "resolved_horizon": None,
                    "horizon_strategy": None,
                    "num_series_loaded": None,
                    "num_series_evaluated": None,
                    "error_message": str(exc),
                }
            )

    run_status_df = pd.DataFrame(run_status_rows)
    run_status_df.to_csv(output_dir / "run_status.csv", index=False)

    if all_per_series_results:
        all_per_series_results_df = pd.concat(all_per_series_results, ignore_index=True)

        per_dataset_summary_df = pd.concat(
            per_dataset_summary_rows,
            ignore_index=True,
        )

        overall_summary_df = (
            all_per_series_results_df.groupby("model", as_index=False)
            .agg(
                n_series=("series_id", "count"),
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                smape=("smape", "mean"),
            )
            .sort_values("mae")
            .reset_index(drop=True)
        )

        all_per_series_results_df.to_csv(
            output_dir / "all_per_series_results.csv",
            index=False,
        )
        per_dataset_summary_df.to_csv(
            output_dir / "per_dataset_summary.csv",
            index=False,
        )
        overall_summary_df.to_csv(
            output_dir / "overall_summary.csv",
            index=False,
        )
    else:
        all_per_series_results_df = pd.DataFrame()
        per_dataset_summary_df = pd.DataFrame()
        overall_summary_df = pd.DataFrame()

    dataset_info_df = build_dataset_summary(
        dataset_keys=dataset_keys,
        base_dir=base_dir,
        fallback_ratio=fallback_ratio,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
    )
    dataset_info_df.to_csv(output_dir / "dataset_info_summary.csv", index=False)

    print("\nBenchmarking complete.")
    print(f"Saved outputs to: {output_dir}")

    return per_dataset_summary_df, overall_summary_df, run_status_df


if __name__ == "__main__":
    per_dataset_summary_df, overall_summary_df, run_status_df = run_naive_benchmarks()

    print("\nRun status:")
    print(run_status_df.head(20))

    print("\nPer-dataset summary:")
    print(per_dataset_summary_df.head(20))

    print("\nOverall summary:")
    print(overall_summary_df)