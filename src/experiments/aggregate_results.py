import os
import pandas as pd


RESULT_DIRS = [
    "results/tables/finance_baselines",
    "results/tables/energy_baselines",
    "results/tables/retail_baselines",
]


def load_all_results() -> pd.DataFrame:
    """
    Load all per-dataset summary CSV files from the domain result folders.
    """
    dfs = []

    for folder in RESULT_DIRS:
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if not file.endswith(".csv"):
                continue

            if file in {
                "all_per_series_results.csv",
                "per_dataset_summary.csv",
                "overall_summary.csv",
                "run_status.csv",
            }:
                continue

            if "summary" not in file:
                continue

            path = os.path.join(folder, file)
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        raise ValueError("No summary CSV files were found in the result folders.")

    return pd.concat(dfs, ignore_index=True)


def create_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per model across all datasets.
    """
    return (
        df.groupby("model")[["mae", "rmse", "smape"]]
        .mean()
        .sort_values("smape")
        .reset_index()
    )


def create_dataset_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best model per dataset using lowest sMAPE.
    """
    idx = df.groupby("dataset_key")["smape"].idxmin()
    return df.loc[idx, ["dataset_key", "model", "smape", "rmse", "mae"]].reset_index(drop=True)


def main() -> None:
    df = load_all_results()

    os.makedirs("results/final", exist_ok=True)

    # Save combined dataset-level summary table
    df.to_csv("results/final/all_results.csv", index=False)

    # Save average metrics by model
    model_comp = create_model_comparison(df)
    model_comp.to_csv("results/final/model_comparison.csv", index=False)

    # Save best model per dataset
    best_models = create_dataset_comparison(df)
    best_models.to_csv("results/final/best_models_per_dataset.csv", index=False)

    print("\nAggregation complete.")
    print("Saved files:")
    print("- results/final/all_results.csv")
    print("- results/final/model_comparison.csv")
    print("- results/final/best_models_per_dataset.csv")


if __name__ == "__main__":
    main()