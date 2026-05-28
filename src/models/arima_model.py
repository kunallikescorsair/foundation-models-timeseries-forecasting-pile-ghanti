"""
ARIMA forecasting baseline.

This module provides:
- a simple ARIMA forecast helper
- a class-based ARIMA model wrapper following the shared model interface
"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from src.models.base_model import BaseForecastModel


def fit_arima_forecast(
    train_values: list[float],
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 0),
) -> np.ndarray:
    """
    Fit ARIMA on one univariate series and forecast future values.

    Falls back to repeating the last value if ARIMA fails.

    Parameters
    ----------
    train_values : list[float]
        Historical training values.
    horizon : int
        Forecast horizon.
    order : tuple[int, int, int]
        ARIMA order (p, d, q).

    Returns
    -------
    np.ndarray
        Forecast of length `horizon`.
    """
    train_arr = np.asarray(train_values, dtype=float)

    if len(train_arr) == 0:
        raise ValueError("train_values must not be empty")

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if len(train_arr) < max(order) + 3:
        return np.full(horizon, train_arr[-1], dtype=float)

    try:
        model = ARIMA(train_arr, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=horizon)
        forecast_arr = np.asarray(forecast, dtype=float)

        if len(forecast_arr) != horizon or not np.all(np.isfinite(forecast_arr)):
            return np.full(horizon, train_arr[-1], dtype=float)

        return forecast_arr

    except Exception:
        return np.full(horizon, train_arr[-1], dtype=float)


class ARIMAModel(BaseForecastModel):
    """
    Class-based ARIMA forecasting model.

    This wrapper follows the shared model interface:
    - fit(train_values)
    - predict(horizon)
    """

    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 0),
    ) -> None:
        self.order = order
        self.train_values: list[float] | None = None

    def fit(self, train_values: list[float]) -> None:
        """
        Store training values for later forecasting.
        """
        if len(train_values) == 0:
            raise ValueError("train_values must not be empty")

        self.train_values = train_values

    def predict(self, horizon: int) -> list[float]:
        """
        Forecast the next `horizon` steps.
        """
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")

        forecast = fit_arima_forecast(
            train_values=self.train_values,
            horizon=horizon,
            order=self.order,
        )
        return forecast.tolist()

    def get_params(self) -> dict[str, Any]:
        """
        Return model parameters for logging.
        """
        return {"order": str(self.order)}