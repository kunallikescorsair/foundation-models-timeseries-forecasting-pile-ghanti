"""
Prophet forecasting baseline.

This module provides:
- a simple Prophet forecast helper
- a class-based Prophet model wrapper following the shared model interface
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecastModel

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False


def infer_prophet_frequency(horizon: int, train_length: int) -> str:
    """
    Infer a practical Prophet frequency string.

    This is a fallback helper for univariate series without explicit timestamps.
    The current implementation defaults to daily frequency because Prophet
    requires a datetime index, and relative spacing is sufficient for baseline use.
    """
    _ = horizon
    _ = train_length
    return "D"


def fit_prophet_forecast(
    train_values: list[float] | np.ndarray,
    horizon: int,
    yearly_seasonality: bool | str = "auto",
    weekly_seasonality: bool | str = "auto",
    daily_seasonality: bool | str = "auto",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0,
    seasonality_mode: str = "additive",
    growth: str = "linear",
) -> np.ndarray:
    """
    Fit Prophet on one univariate series and forecast future values.

    Falls back to repeating the last value if Prophet fails.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed.")

    train_arr = np.asarray(train_values, dtype=float).reshape(-1)

    if train_arr.size == 0:
        raise ValueError("train_values must not be empty")

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if not np.all(np.isfinite(train_arr)):
        mean_val = np.nanmean(train_arr)
        if not np.isfinite(mean_val):
            mean_val = 0.0

        train_arr = np.nan_to_num(
            train_arr,
            nan=float(mean_val),
            posinf=float(mean_val),
            neginf=float(mean_val),
        )

    fallback = np.full(horizon, train_arr[-1], dtype=float)

    try:
        freq = infer_prophet_frequency(horizon=horizon, train_length=len(train_arr))

        history_df = pd.DataFrame(
            {
                "ds": pd.date_range(
                    start="2000-01-01",
                    periods=len(train_arr),
                    freq=freq,
                ),
                "y": train_arr,
            }
        )

        model = Prophet(
            growth=growth,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
        )

        model.fit(history_df)

        future_df = model.make_future_dataframe(
            periods=horizon,
            freq=freq,
            include_history=True,
        )

        forecast_df = model.predict(future_df)

        yhat = forecast_df["yhat"].iloc[-horizon:].to_numpy(dtype=float)

        if len(yhat) != horizon or not np.all(np.isfinite(yhat)):
            return fallback

        return yhat

    except Exception:
        return fallback


class ProphetModel(BaseForecastModel):
    """
    Class-based Prophet forecasting model.

    This wrapper follows the shared model interface:
    - fit(train_values)
    - predict(horizon)
    """

    name = "prophet"

    def __init__(
        self,
        yearly_seasonality: bool | str = "auto",
        weekly_seasonality: bool | str = "auto",
        daily_seasonality: bool | str = "auto",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        growth: str = "linear",
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.growth = growth

        self.train_values: list[float] | np.ndarray | None = None

    def fit(self, train_values: list[float] | np.ndarray) -> None:
        """
        Store training values for later forecasting.
        """
        train_arr = np.asarray(train_values, dtype=float).reshape(-1)

        if train_arr.size == 0:
            raise ValueError("train_values must not be empty")

        self.train_values = train_arr

    def predict(self, horizon: int) -> list[float]:
        """
        Forecast the next `horizon` steps.
        """
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")

        forecast = fit_prophet_forecast(
            train_values=self.train_values,
            horizon=horizon,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            growth=self.growth,
        )

        return forecast.tolist()

    def get_params(self) -> dict[str, Any]:
        """
        Return model parameters for logging.
        """
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "seasonality_mode": self.seasonality_mode,
            "growth": self.growth,
        }