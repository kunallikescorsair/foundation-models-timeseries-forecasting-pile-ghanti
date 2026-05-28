"""
Dataset loader utilities for Monash forecasting datasets from Time Series PILE.

This module provides reusable functions to:
- discover all Monash .tsf datasets
- normalize dataset keys and metadata
- resolve forecast horizon robustly
- load one dataset into a standardized bundle
- summarize available datasets
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.tsf_parser import (
    build_split_dataframe,
    convert_to_long_format,
    parse_tsf_file,
)


def get_default_monash_base_dir() -> Path:
    """
    Resolve the default Monash dataset directory.

    Priority:
    1. MONASH_TSF_BASE_DIR environment variable
    2. Hugging Face cache under the user's home directory

    Returns
    -------
    Path
        Resolved Monash dataset directory.

    Raises
    ------
    FileNotFoundError
        If no valid Monash dataset directory is found.
    """
    env_path = os.getenv("MONASH_TSF_BASE_DIR")
    if env_path:
        env_dir = Path(env_path).expanduser().resolve()
        if env_dir.exists():
            return env_dir

    hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    candidate_dirs = sorted(
        hf_root.glob(
            "datasets--AutonLab--Timeseries-PILE/snapshots/*/forecasting/monash"
        )
    )

    for candidate in reversed(candidate_dirs):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate Monash TSF directory. "
        "Set MONASH_TSF_BASE_DIR or download the dataset into the Hugging Face cache."
    )


DEFAULT_MONASH_BASE_DIR = get_default_monash_base_dir()


def normalize_dataset_key(file_name: str) -> str:
    """
    Convert a Monash dataset file name into a normalized dataset key.

    Examples
    --------
    bitcoin_dataset_without_missing_values.tsf -> bitcoin
    m4_quarterly_dataset.tsf -> m4_quarterly
    solar_10_minutes_dataset.tsf -> solar_10_minutes
    """
    dataset_key = file_name.replace(".tsf", "")
    dataset_key = dataset_key.replace("_dataset_without_missing_values", "")
    dataset_key = dataset_key.replace("_dataset", "")
    return dataset_key


def discover_monash_datasets(
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR,
) -> dict[str, Path]:
    """
    Discover all Monash .tsf datasets in the given directory.

    Parameters
    ----------
    base_dir : str | Path
        Directory containing Monash .tsf files.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset_key -> full file path.
    """
    base_dir = Path(base_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    dataset_map: dict[str, Path] = {}

    for file_path in sorted(base_dir.glob("*.tsf")):
        if file_path.name.startswith("."):
            continue

        dataset_key = normalize_dataset_key(file_path.name)
        dataset_map[dataset_key] = file_path

    if not dataset_map:
        raise FileNotFoundError(f"No .tsf files found in directory: {base_dir}")

    return dataset_map


def list_available_datasets(
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR,
) -> list[str]:
    """
    Return all available Monash dataset keys.
    """
    return sorted(discover_monash_datasets(base_dir).keys())


def get_dataset_path(
    dataset_key: str,
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR,
) -> Path:
    """
    Resolve the file path for a dataset key.

    Parameters
    ----------
    dataset_key : str
        Normalized dataset key.
    base_dir : str | Path
        Monash dataset directory.

    Returns
    -------
    Path
        Full file path to the .tsf file.
    """
    dataset_map = discover_monash_datasets(base_dir)

    if dataset_key not in dataset_map:
        raise ValueError(
            f"Unknown dataset_key: {dataset_key}. "
            f"Available datasets: {sorted(dataset_map.keys())}"
        )

    return dataset_map[dataset_key]


def normalize_frequency_label(freq_text: str | None) -> str | None:
    """
    Normalize TSF frequency text to a stable label.

    Parameters
    ----------
    freq_text : str | None
        Raw frequency text from metadata.

    Returns
    -------
    str | None
        Normalized frequency label.
    """
    if freq_text is None:
        return None

    freq = freq_text.strip().lower().replace("-", "_").replace(" ", "_")

    mapping = {
        "yearly": "yearly",
        "annual": "yearly",
        "quarterly": "quarterly",
        "monthly": "monthly",
        "weekly": "weekly",
        "daily": "daily",
        "hourly": "hourly",
        "half_hourly": "half_hourly",
        "30_minutes": "half_hourly",
        "minutely": "minutely",
        "minute": "minutely",
        "10_minutes": "10_minutes",
        "4_seconds": "4_seconds",
        "seconds": "seconds",
        "secondly": "seconds",
    }

    return mapping.get(freq, freq)


def infer_domain_from_dataset_key(dataset_key: str) -> str:
    """
    Infer a coarse domain label from the dataset key.

    Parameters
    ----------
    dataset_key : str
        Normalized dataset key.

    Returns
    -------
    str
        Domain label.
    """
    key = dataset_key.lower()

    finance_keywords = {"bitcoin", "fred", "m1", "m3", "m4", "sunspot"}
    energy_keywords = {
        "electricity",
        "solar",
        "wind",
        "weather",
        "temperature",
        "australian_electricity_demand",
        "london_smart_meters",
        "saugeenday",
    }
    retail_keywords = {"dominick", "nn5", "car_parts"}
    traffic_keywords = {"traffic", "pedestrian", "rideshare", "vehicle_trips"}
    health_keywords = {"hospital", "covid_deaths", "us_births"}
    tourism_keywords = {"tourism"}
    web_keywords = {"web_traffic"}
    environment_keywords = {"saugeenday"}

    if any(token in key for token in finance_keywords):
        return "finance"
    if any(token in key for token in energy_keywords):
        return "energy"
    if any(token in key for token in retail_keywords):
        return "retail"
    if any(token in key for token in traffic_keywords):
        return "transport"
    if any(token in key for token in health_keywords):
        return "health"
    if any(token in key for token in tourism_keywords):
        return "tourism"
    if any(token in key for token in web_keywords):
        return "web"
    if any(token in key for token in environment_keywords):
        return "environment"

    return "unknown"


def resolve_forecast_horizon(
    metadata: dict[str, Any],
    records_df: pd.DataFrame,
    normalized_frequency: str | None = None,
    fallback_ratio: float = 0.1,
    min_horizon: int = 1,
    max_horizon: int | None = None,
) -> tuple[int, str]:
    """
    Resolve the forecast horizon using metadata or a capped fallback rule.

    Parameters
    ----------
    metadata : dict[str, Any]
        Parsed TSF metadata.
    records_df : pd.DataFrame
        Parsed dataset records.
    normalized_frequency : str | None
        Normalized frequency label.
    fallback_ratio : float
        Ratio of the shortest series length used for fallback horizon.
    min_horizon : int
        Minimum allowed fallback horizon.
    max_horizon : int | None
        Optional global cap for fallback horizon.

    Returns
    -------
    tuple[int, str]
        Resolved horizon and the strategy used.
    """
    metadata_horizon = metadata.get("horizon")

    # Preserve dataset metadata horizon as-is if valid.
    if metadata_horizon is not None:
        try:
            metadata_horizon = int(metadata_horizon)
            if metadata_horizon > 0:
                return metadata_horizon, "metadata"
        except (TypeError, ValueError):
            pass

    if records_df.empty:
        raise ValueError("Cannot resolve horizon from an empty records dataframe")

    series_lengths = records_df["series_values"].apply(len)
    shortest_length = int(series_lengths.min())

    fallback_horizon = max(min_horizon, int(shortest_length * fallback_ratio))

    frequency_cap = get_frequency_horizon_cap(normalized_frequency)
    fallback_horizon = min(fallback_horizon, frequency_cap)

    if max_horizon is not None:
        fallback_horizon = min(fallback_horizon, max_horizon)

    # Ensure at least one training point remains for the shortest series.
    fallback_horizon = min(fallback_horizon, max(1, shortest_length - 1))

    if fallback_horizon <= 0:
        raise ValueError("Resolved forecast horizon is not valid")

    return fallback_horizon, "fallback_capped"


def ensure_series_identifier(records_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the records dataframe contains a stable series_name column.

    Parameters
    ----------
    records_df : pd.DataFrame
        Parsed records dataframe.

    Returns
    -------
    pd.DataFrame
        Records dataframe with a valid series_name column.
    """
    records_df = records_df.copy()

    if "series_name" not in records_df.columns:
        records_df["series_name"] = [
            f"series_{idx:06d}" for idx in range(1, len(records_df) + 1)
        ]
        return records_df

    generated_names: list[str] = []
    used_names: set[str] = set()

    for idx, raw_name in enumerate(records_df["series_name"], start=1):
        if raw_name is None or str(raw_name).strip() == "":
            candidate = f"series_{idx:06d}"
        else:
            candidate = str(raw_name).strip()

        if candidate in used_names:
            candidate = f"{candidate}_{idx:06d}"

        used_names.add(candidate)
        generated_names.append(candidate)

    records_df["series_name"] = generated_names
    return records_df


def load_monash_dataset(
    dataset_key: str,
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR,
    fallback_ratio: float = 0.1,
    min_horizon: int = 1,
    max_horizon: int | None = None,
) -> dict[str, Any]:
    """
    Load one Monash dataset and return a standardized bundle.

    Parameters
    ----------
    dataset_key : str
        Normalized dataset key.
    base_dir : str | Path
        Monash dataset directory.
    fallback_ratio : float
        Ratio used when metadata horizon is missing.
    min_horizon : int
        Minimum fallback horizon.
    max_horizon : int | None
        Optional cap for fallback horizon.

    Returns
    -------
    dict[str, Any]
        Standardized dataset bundle.
    """
    file_path = get_dataset_path(dataset_key, base_dir)

    metadata, attribute_defs, records_df = parse_tsf_file(file_path)
    records_df = ensure_series_identifier(records_df)

    normalized_frequency = normalize_frequency_label(metadata.get("frequency"))
    resolved_horizon, horizon_strategy = resolve_forecast_horizon(
        metadata=metadata,
        records_df=records_df,
        normalized_frequency=normalized_frequency,
        fallback_ratio=fallback_ratio,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
    )

    long_df = convert_to_long_format(records_df, metadata)
    split_df = build_split_dataframe(records_df, resolved_horizon)

    bundle = {
        "dataset_key": dataset_key,
        "domain": infer_domain_from_dataset_key(dataset_key),
        "file_path": file_path,
        "file_name": f"{dataset_key}.tsf",
        "metadata": metadata,
        "attribute_defs": attribute_defs,
        "records_df": records_df,
        "long_df": long_df,
        "split_df": split_df,
        "normalized_frequency": normalized_frequency,
        "resolved_horizon": resolved_horizon,
        "horizon_strategy": horizon_strategy,
    }

    return bundle


def build_dataset_summary(
    dataset_keys: list[str] | None = None,
    base_dir: str | Path = DEFAULT_MONASH_BASE_DIR,
    fallback_ratio: float = 0.1,
    min_horizon: int = 1,
    max_horizon: int | None = None,
) -> pd.DataFrame:
    """
    Build a summary table for Monash datasets.

    Parameters
    ----------
    dataset_keys : list[str] | None
        Dataset keys to summarize. If None, summarize all discovered datasets.
    base_dir : str | Path
        Monash dataset directory.
    fallback_ratio : float
        Ratio used when metadata horizon is missing.
    min_horizon : int
        Minimum fallback horizon.
    max_horizon : int | None
        Optional cap for fallback horizon.

    Returns
    -------
    pd.DataFrame
        Dataset summary dataframe.
    """
    if dataset_keys is None:
        dataset_keys = list_available_datasets(base_dir)

    rows: list[dict[str, Any]] = []

    for dataset_key in dataset_keys:
        try:
            bundle = load_monash_dataset(
                dataset_key=dataset_key,
                base_dir=base_dir,
                fallback_ratio=fallback_ratio,
                min_horizon=min_horizon,
                max_horizon=max_horizon,
            )

            rows.append(
                {
                    "dataset_key": dataset_key,
                    "domain": bundle["domain"],
                    "file_name": bundle["file_path"],
                    "raw_frequency": bundle["metadata"].get("frequency"),
                    "normalized_frequency": bundle["normalized_frequency"],
                    "metadata_horizon": bundle["metadata"].get("horizon"),
                    "resolved_horizon": bundle["resolved_horizon"],
                    "horizon_strategy": bundle["horizon_strategy"],
                    "missing": bundle["metadata"].get("missing"),
                    "equallength": bundle["metadata"].get("equallength"),
                    "num_series": len(bundle["records_df"]),
                    "num_long_rows": len(bundle["long_df"]),
                    "num_split_series": len(bundle["split_df"]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "domain": infer_domain_from_dataset_key(dataset_key),
                    "file_name": get_dataset_path(dataset_key, base_dir).name,
                    "raw_frequency": None,
                    "normalized_frequency": None,
                    "metadata_horizon": None,
                    "resolved_horizon": None,
                    "horizon_strategy": None,
                    "missing": None,
                    "equallength": None,
                    "num_series": None,
                    "num_long_rows": None,
                    "num_split_series": None,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows).sort_values("dataset_key").reset_index(drop=True)

def get_frequency_horizon_cap(normalized_frequency: str | None) -> int:
    """
    Return a practical maximum fallback horizon for a given normalized frequency.

    These caps are only used when dataset metadata does not provide a horizon.
    They are intended to keep experiments computationally reasonable while still
    allowing meaningful forecasting windows.

    Parameters
    ----------
    normalized_frequency : str | None
        Normalized frequency label.

    Returns
    -------
    int
        Maximum fallback horizon.
    """
    cap_map = {
        "yearly": 8,
        "quarterly": 8,
        "monthly": 24,
        "weekly": 26,
        "daily": 30,
        "hourly": 168,
        "half_hourly": 336,
        "10_minutes": 144,
        "minutely": 120,
        "4_seconds": 60,
        "seconds": 60,
    }

    return cap_map.get(normalized_frequency, 24)