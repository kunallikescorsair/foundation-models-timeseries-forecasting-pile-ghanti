"""
TSF parser utilities for Monash-style forecasting datasets from Time Series PILE.

This module provides reusable functions to:
- parse .tsf files
- extract metadata and attribute definitions
- convert parsed records into pandas DataFrames
- expand series into long format
- split each series into train/test using forecast horizon
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_frequency(freq_text: str | None) -> str | None:
    """
    Map TSF frequency text to a pandas-compatible frequency string.

    Parameters
    ----------
    freq_text : str | None
        Frequency text from TSF metadata, e.g. 'yearly', 'monthly'.

    Returns
    -------
    str | None
        Pandas frequency string if recognized, otherwise None.
    """
    if freq_text is None:
        return None

    freq_text = freq_text.strip().lower()

    mapping = {
        "yearly": "YS",
        "annual": "YS",
        "quarterly": "QS",
        "monthly": "MS",
        "weekly": "W",
        "daily": "D",
        "hourly": "H",
        "minutely": "T",
        "10_minutes": "10T",
        "half_hourly": "30T",
        "4_seconds": "4S",
        "seconds": "S",
    }

    return mapping.get(freq_text, None)


def parse_bool(text: str) -> bool | str:
    """
    Parse TSF boolean metadata fields.

    Parameters
    ----------
    text : str
        Raw metadata value.

    Returns
    -------
    bool | str
        Boolean if recognized, otherwise the original string.
    """
    text = text.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return text


def parse_scalar_value(raw_value: str, declared_type: str) -> Any:
    """
    Parse one scalar value according to its declared TSF type.

    Parameters
    ----------
    raw_value : str
        Raw value from TSF row.
    declared_type : str
        Declared TSF attribute type.

    Returns
    -------
    Any
        Parsed Python value.
    """
    if raw_value == "?":
        return None

    declared_type = declared_type.lower().strip()

    if declared_type == "numeric":
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    if declared_type == "integer":
        try:
            return int(raw_value)
        except ValueError:
            return raw_value

    if declared_type == "date":
        for fmt in ("%Y-%m-%d %H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw_value, fmt)
            except ValueError:
                continue
        return raw_value

    return raw_value


def parse_series_values(series_text: str) -> list[float]:
    """
    Parse comma-separated time series values into a list of floats.

    Missing values represented by '?' are converted to np.nan.

    Parameters
    ----------
    series_text : str
        Comma-separated series values.

    Returns
    -------
    list[float]
        Parsed numeric series.
    """
    values: list[float] = []

    for item in series_text.split(","):
        item = item.strip()

        if item == "?":
            values.append(np.nan)
        else:
            try:
                values.append(float(item))
            except ValueError:
                values.append(np.nan)

    return values


def parse_tsf_file(file_path: str | Path) -> tuple[dict[str, Any], list[tuple[str, str]], pd.DataFrame]:
    """
    Parse a Monash-style .tsf file.

    Parameters
    ----------
    file_path : str | Path
        Path to the TSF file.

    Returns
    -------
    metadata : dict
        File-level metadata such as relation, frequency, horizon, missing, equallength.
    attribute_defs : list[tuple[str, str]]
        Attribute definitions in order of declaration.
    records_df : pd.DataFrame
        One row per time series with metadata columns and series_values.
    """
    file_path = Path(file_path)

    metadata: dict[str, Any] = {}
    attribute_defs: list[tuple[str, str]] = []
    records: list[dict[str, Any]] = []

    in_data_section = False

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if not in_data_section:
                lower_line = line.lower()

                if lower_line.startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 3:
                        attr_name = parts[1].strip()
                        attr_type = parts[2].strip()
                        attribute_defs.append((attr_name, attr_type))
                    continue

                if lower_line.startswith("@frequency"):
                    metadata["frequency"] = line.split(maxsplit=1)[1].strip()
                    continue

                if lower_line.startswith("@horizon"):
                    metadata["horizon"] = int(line.split(maxsplit=1)[1].strip())
                    continue

                if lower_line.startswith("@missing"):
                    metadata["missing"] = parse_bool(line.split(maxsplit=1)[1].strip())
                    continue

                if lower_line.startswith("@equallength"):
                    metadata["equallength"] = parse_bool(line.split(maxsplit=1)[1].strip())
                    continue

                if lower_line.startswith("@relation"):
                    metadata["relation"] = line.split(maxsplit=1)[1].strip()
                    continue

                if lower_line == "@data":
                    in_data_section = True
                    continue

            else:
                expected_prefix_parts = len(attribute_defs)
                parts = line.split(":", maxsplit=expected_prefix_parts)

                if len(parts) != expected_prefix_parts + 1:
                    raise ValueError(
                        f"Unexpected row format in file {file_path.name}: {line[:200]}"
                    )

                row: dict[str, Any] = {}

                for i, (attr_name, attr_type) in enumerate(attribute_defs):
                    row[attr_name] = parse_scalar_value(parts[i], attr_type)

                row["series_values"] = parse_series_values(parts[-1])
                row["series_length"] = len(row["series_values"])

                records.append(row)

    records_df = pd.DataFrame(records)
    return metadata, attribute_defs, records_df


def convert_to_long_format(records_df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """
    Convert parsed TSF records into long format.

    Output columns:
    - series_id
    - timestamp
    - step_index
    - value

    Parameters
    ----------
    records_df : pd.DataFrame
        Parsed records dataframe from parse_tsf_file.
    metadata : dict
        Metadata returned by parse_tsf_file.

    Returns
    -------
    pd.DataFrame
        Long-format time series dataframe.
    """
    pandas_freq = parse_frequency(metadata.get("frequency"))

    long_rows: list[dict[str, Any]] = []

    for _, row in records_df.iterrows():
        series_id = row.get("series_name", None)
        start_timestamp = row.get("start_timestamp", None)
        values = row["series_values"]

        timestamps: list[Any]
        if isinstance(start_timestamp, datetime) and pandas_freq is not None:
            try:
                timestamps = list(
                    pd.date_range(
                        start=start_timestamp,
                        periods=len(values),
                        freq=pandas_freq,
                    )
                )
            except Exception:
                timestamps = [None] * len(values)
        else:
            timestamps = [None] * len(values)

        for idx, value in enumerate(values):
            long_rows.append(
                {
                    "series_id": series_id,
                    "timestamp": timestamps[idx],
                    "step_index": idx,
                    "value": value,
                }
            )

    return pd.DataFrame(long_rows)


def split_series_by_horizon(series_values: list[float], horizon: int) -> tuple[list[float] | None, list[float] | None]:
    """
    Split one series into train and test using the forecast horizon.

    Parameters
    ----------
    series_values : list[float]
        Full time series values.
    horizon : int
        Forecast horizon.

    Returns
    -------
    tuple[list[float] | None, list[float] | None]
        Train values and test values.
        Returns (None, None) if the series is too short.
    """
    if len(series_values) <= horizon:
        return None, None

    train_values = series_values[:-horizon]
    test_values = series_values[-horizon:]

    return train_values, test_values


def build_split_dataframe(records_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Build a train/test split dataframe for all parsed series.

    Parameters
    ----------
    records_df : pd.DataFrame
        Parsed records dataframe.
    horizon : int
        Forecast horizon.

    Returns
    -------
    pd.DataFrame
        Dataframe with train/test lists and lengths for each series.
    """
    split_rows: list[dict[str, Any]] = []

    for _, row in records_df.iterrows():
        train_values, test_values = split_series_by_horizon(row["series_values"], horizon)

        if train_values is None:
            continue

        split_rows.append(
            {
                "series_id": row.get("series_name"),
                "train_values": train_values,
                "test_values": test_values,
                "train_length": len(train_values),
                "test_length": len(test_values),
            }
        )

    return pd.DataFrame(split_rows)