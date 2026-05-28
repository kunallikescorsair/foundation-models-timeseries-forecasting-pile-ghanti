"""
TinyTimeMixer (TTM) foundation model wrapper.

This wrapper uses IBM Granite TSFM utilities for zero-shot forecasting and
fits the shared project interface:
- fit(train_values)
- predict(horizon)
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch

from src.models.base_model import BaseForecastModel

try:
    from tsfm_public import TimeSeriesPreprocessor
    from tsfm_public.toolkit.get_model import get_model
    from tsfm_public.toolkit.time_series_forecasting_pipeline import (
        TimeSeriesForecastingPipeline,
    )

    TTM_AVAILABLE = True
except ImportError:
    TimeSeriesPreprocessor = None
    get_model = None
    TimeSeriesForecastingPipeline = None
    TTM_AVAILABLE = False


def map_project_frequency_to_ttm_freq(normalized_frequency: str | None) -> str:
    """
    Map project frequency labels to TTM frequency tokens.

    Parameters
    ----------
    normalized_frequency : str | None
        Project-normalized frequency label.

    Returns
    -------
    str
        Frequency token expected by granite-tsfm.

    Raises
    ------
    ValueError
        If the frequency is not currently supported by the wrapper.
    """
    mapping = {
        "10_minutes": "10min",
        "half_hourly": "30min",
        "hourly": "1h",
        "daily": "1d",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",  
        "yearly": "Y", 
    }

    if normalized_frequency not in mapping:
        raise ValueError(
            f"TTM frequency mapping not defined for: {normalized_frequency}"
        )

    return mapping[normalized_frequency]


def map_ttm_freq_to_pandas_freq(ttm_freq: str) -> str:
    """
    Convert TTM frequency token to pandas frequency string.
    """
    mapping = {
        "10min": "10min",
        "30min": "30min",
        "1h": "1h",
        "1d": "1D",
        "W": "W",
        "M": "M",
        "Q": "Q",   
        "Y": "Y",   
    }

    if ttm_freq not in mapping:
        raise ValueError(f"Unsupported TTM frequency token: {ttm_freq}")

    return mapping[ttm_freq]


class TTMModel(BaseForecastModel):
    """
    Zero-shot TinyTimeMixer model wrapper.

    Notes
    -----
    - Uses IBM's `get_model()` utility to select a suitable pre-trained TTM.
    - Uses synthetic timestamps because the benchmark pipeline operates on
      value sequences rather than raw calendar-aware dataframes.
    - Intended for supported frequencies only.
    """

    name = "ttm"

    _shared_models: ClassVar[dict[tuple, Any]] = {}

    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
        normalized_frequency: str | None = None,
        max_context: int = 512,
        prefer_longer_context: bool = True,
        prefer_l1_loss: bool = False,
    ) -> None:
        self.model_path = model_path
        self.normalized_frequency = normalized_frequency
        self.max_context = max_context
        self.prefer_longer_context = prefer_longer_context
        self.prefer_l1_loss = prefer_l1_loss

        self.train_values: list[float] | None = None

    @classmethod
    def _get_or_load_model(
        cls,
        model_path: str,
        normalized_frequency: str,
        context_length: int,
        prediction_length: int,
        prefer_longer_context: bool,
        prefer_l1_loss: bool,
    ) -> Any:
        """
        Load and cache a pre-trained TTM selected via IBM's get_model utility.
        """
        if not TTM_AVAILABLE:
            raise ImportError(
                "granite-tsfm is not installed. Run `pip install granite-tsfm`."
            )

        freq = map_project_frequency_to_ttm_freq(normalized_frequency)

        key = (
            model_path,
            normalized_frequency,
            context_length,
            prediction_length,
            prefer_longer_context,
            prefer_l1_loss,
        )

        if key in cls._shared_models:
            return cls._shared_models[key]

        model = get_model(
            model_path=model_path,
            model_name="ttm",
            context_length=context_length,
            prediction_length=prediction_length,
            freq=freq,
            prefer_longer_context=prefer_longer_context,
            prefer_l1_loss=prefer_l1_loss,
        )

        cls._shared_models[key] = model
        return model

    def fit(self, train_values: list[float]) -> None:
        """
        Store training values for zero-shot inference.
        """
        if len(train_values) == 0:
            raise ValueError("train_values must not be empty")

        self.train_values = train_values

    def predict(self, horizon: int) -> list[float]:
        """
        Produce a zero-shot TTM forecast.

        Parameters
        ----------
        horizon : int
            Forecast horizon.

        Returns
        -------
        list[float]
            Forecast values.
        """
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if self.normalized_frequency is None:
            raise ValueError("TTMModel requires normalized_frequency")

        if not TTM_AVAILABLE:
            raise ImportError(
                "granite-tsfm is not installed. Run `pip install granite-tsfm`."
            )

        train_arr = np.asarray(self.train_values, dtype=float)

        # Keep only recent context.
        context_length = min(len(train_arr), self.max_context)

        if context_length < 8:
            last_value = float(train_arr[-1])
            return [last_value] * horizon

        train_arr = train_arr[-context_length:]
        train_arr = np.nan_to_num(train_arr, nan=float(np.nanmean(train_arr)))

        model = self._get_or_load_model(
            model_path=self.model_path,
            normalized_frequency=self.normalized_frequency,
            context_length=context_length,
            prediction_length=horizon,
            prefer_longer_context=self.prefer_longer_context,
            prefer_l1_loss=self.prefer_l1_loss,
        )

        ttm_freq = map_project_frequency_to_ttm_freq(self.normalized_frequency)
        pandas_freq = map_ttm_freq_to_pandas_freq(ttm_freq)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2000-01-01",
                    periods=len(train_arr),
                    freq=pandas_freq,
                ),
                "value": train_arr,
            }
        )

        # Force CPU for TTM because MPS is unstable for this model on macOS.
        original_mps_is_available = torch.backends.mps.is_available
        original_mps_is_built = torch.backends.mps.is_built

        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

        try:
            tsp = TimeSeriesPreprocessor(
                timestamp_column="timestamp",
                id_columns=[],
                target_columns=["value"],
                context_length=model.config.context_length,
                prediction_length=horizon,
                freq=ttm_freq,
                scaling=True,
            )
            tsp.train(df)

            forecast_pipeline = TimeSeriesForecastingPipeline(
                model=model,
                timestamp_column="timestamp",
                id_columns=[],
                target_columns=["value"],
                freq=ttm_freq,
                feature_extractor=tsp,
                explode_forecasts=False,
                inverse_scale_outputs=True,
                batch_size=1,
            )

            forecasts = forecast_pipeline(df)

        finally:
            torch.backends.mps.is_available = original_mps_is_available
            torch.backends.mps.is_built = original_mps_is_built

        pred_col = "value_prediction"
        if pred_col not in forecasts.columns:
            raise ValueError(
                f"Expected prediction column '{pred_col}' not found. "
                f"Available columns: {list(forecasts.columns)}"
            )

        pred = forecasts[pred_col].iloc[0]
        pred = np.asarray(pred, dtype=float).reshape(-1)

        if len(pred) > horizon:
            pred = pred[:horizon]

        if len(pred) < horizon:
            pad_value = float(pred[-1]) if len(pred) > 0 else float(train_arr[-1])
            padded = np.full(shape=horizon, fill_value=pad_value, dtype=float)
            padded[: len(pred)] = pred
            pred = padded

        if not np.all(np.isfinite(pred)):
            last_value = float(train_arr[-1])
            return [last_value] * horizon

        return pred.tolist()

    def get_params(self) -> dict[str, Any]:
        """
        Return model parameters for logging.
        """
        return {
            "model_path": self.model_path,
            "normalized_frequency": self.normalized_frequency,
            "max_context": self.max_context,
            "prefer_longer_context": self.prefer_longer_context,
            "prefer_l1_loss": self.prefer_l1_loss,
        }