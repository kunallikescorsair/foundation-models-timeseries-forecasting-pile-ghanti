"""
Dataset loader utilities for selected Monash forecasting datasets
from the Time Series PILE repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.tsf_parser import (
    parse_tsf_file,
    convert_to_long_format,
    build_split_dataframe,
)


# Base path to downloaded forecasting datasets
DEFAULT_MONASH_BASE_DIR = Path(
    "/Users/kunalgurung/.cache/huggingface/hub/datasets--AutonLab--Timeseries-PILE/snapshots/ea89753da2b451928436adb333c7a2e892461c7d/forecasting/monash"
)


# Selected datasets for the capstone project
SELECTED_DATASETS = {
    # Economics
    "m3_yearly": "m3_yearly_dataset.tsf",
    "m3_quarterly": "m3_quarterly_dataset.tsf",
    "m3_monthly": "m3_monthly_dataset.tsf",

    # Tourism
    "tourism_yearly": "tourism_yearly_dataset.tsf",
    "tourism_quarterly": "tourism_quarterly_dataset.tsf",
    "tourism_monthly": "tourism_monthly_dataset.tsf",

    # Energy
    "electricity_weekly": "electricity_weekly_dataset.tsf",
    "solar_weekly": "solar_weekly_dataset.tsf",
}


DOMAIN_MAP = {
    "m3_yearly": "economics",
    "m3_quarterly": "economics",
    "m3_monthly": "economics",

    "tourism_yearly": "tourism",
    "tourism_quarterly": "tourism",
    "tourism_monthly": "tourism",

    "electricity_weekly": "energy",
    "solar_weekly": "energy",
}


def list_available_datasets() -> list[str]:
    """
    Return the list of supported dataset keys.
    """
    return sorted(SELECTED_DATASETS.keys())


def get_dataset_path(
    dataset_key: str,
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR
) -> Path:
    """
    Resolve the file path for a selected dataset.

    Parameters
    ----------
    dataset_key : str
        Short dataset key, e.g. 'm3_monthly'.
    base_dir : str | Path
        Base directory containing Monash .tsf files.

    Returns
    -------
    Path
        Full path to the dataset file.
    """
    if dataset_key not in SELECTED_DATASETS:
        raise ValueError(
            f"Unknown dataset_key: {dataset_key}. "
            f"Available datasets: {list_available_datasets()}"
        )

    base_dir = Path(base_dir)
    file_name = SELECTED_DATASETS[dataset_key]
    file_path = base_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return file_path


def load_monash_dataset(
    dataset_key: str,
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR
) -> dict[str, Any]:
    """
    Load one selected Monash dataset and return a standardized bundle.

    Parameters
    ----------
    dataset_key : str
        Short dataset key, e.g. 'm3_monthly'.
    base_dir : str | Path
        Base directory containing Monash .tsf files.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - dataset_key
        - domain
        - file_path
        - metadata
        - attribute_defs
        - records_df
        - long_df
        - split_df
    """
    file_path = get_dataset_path(dataset_key, base_dir)

    metadata, attribute_defs, records_df = parse_tsf_file(file_path)
    long_df = convert_to_long_format(records_df, metadata)
    split_df = build_split_dataframe(records_df, metadata["horizon"])

    bundle = {
        "dataset_key": dataset_key,
        "domain": DOMAIN_MAP.get(dataset_key, "unknown"),
        "file_path": file_path,
        "metadata": metadata,
        "attribute_defs": attribute_defs,
        "records_df": records_df,
        "long_df": long_df,
        "split_df": split_df,
    }

    return bundle


def build_dataset_summary(
    dataset_keys: list[str] | None = None,
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR
) -> pd.DataFrame:
    """
    Build a summary table for selected datasets.

    Parameters
    ----------
    dataset_keys : list[str] | None
        Dataset keys to summarize. If None, summarize all supported datasets.
    base_dir : str | Path
        Base directory containing Monash .tsf files.

    Returns
    -------
    pd.DataFrame
        Summary dataframe for selected datasets.
    """
    if dataset_keys is None:
        dataset_keys = list_available_datasets()

    rows = []

    for dataset_key in dataset_keys:
        bundle = load_monash_dataset(dataset_key, base_dir=base_dir)

        rows.append({
            "dataset_key": dataset_key,
            "domain": bundle["domain"],
            "file_name": bundle["file_path"].name,
            "frequency": bundle["metadata"].get("frequency"),
            "horizon": bundle["metadata"].get("horizon"),
            "missing": bundle["metadata"].get("missing"),
            "equallength": bundle["metadata"].get("equallength"),
            "num_series": len(bundle["records_df"]),
            "num_long_rows": len(bundle["long_df"]),
        })

    return pd.DataFrame(rows)