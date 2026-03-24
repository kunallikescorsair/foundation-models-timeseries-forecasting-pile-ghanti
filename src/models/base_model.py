"""
Base interface for all forecasting models.

Every model in this project should implement:
- fit(train_values)
- predict(horizon)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseForecastModel(ABC):
    """
    Abstract base class for forecasting models.
    """

    name: str = "base_model"

    @abstractmethod
    def fit(self, train_values: list[float]) -> None:
        """
        Fit the model on one univariate training series.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> list[float]:
        """
        Predict the next `horizon` time steps.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        Return model parameters for logging.
        """
        return {}