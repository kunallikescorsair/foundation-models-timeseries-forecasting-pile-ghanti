"""
LSTM forecasting baseline.

This module provides a simple univariate LSTM forecasting model
that follows the shared model interface:
- fit(train_values)
- predict(horizon)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseForecastModel


class SimpleLSTMNetwork(nn.Module):
    """
    Minimal LSTM network for direct multi-step forecasting.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, seq_len, input_size)

        Returns
        -------
        torch.Tensor
            Shape: (batch_size, output_size)
        """
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction


def create_lstm_windows(
    series_values: list[float],
    input_window: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create supervised windows for LSTM training.

    Parameters
    ----------
    series_values : list[float]
        Input time series.
    input_window : int
        Number of past values used as input.
    forecast_horizon : int
        Number of future values to predict.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X and y arrays.
    """
    values = np.asarray(series_values, dtype=np.float32)

    X, y = [], []

    max_start = len(values) - input_window - forecast_horizon + 1
    for start_idx in range(max_start):
        end_input = start_idx + input_window
        end_target = end_input + forecast_horizon

        X.append(values[start_idx:end_input])
        y.append(values[end_input:end_target])

    if not X:
        return (
            np.empty((0, input_window), dtype=np.float32),
            np.empty((0, forecast_horizon), dtype=np.float32),
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class LSTMModel(BaseForecastModel):
    """
    Simple univariate LSTM baseline.

    This model trains on one series at a time and predicts the full
    forecast horizon directly.
    """

    name = "lstm"

    def __init__(
        self,
        input_window: int = 24,
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str | None = None,
        random_seed: int = 42,
    ) -> None:
        self.input_window = input_window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.train_values: list[float] | None = None
        self.model: SimpleLSTMNetwork | None = None
        self.forecast_horizon_: int | None = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def fit(self, train_values: list[float]) -> None:
        """
        Store train values only.

        Actual training happens inside predict(), because forecast horizon
        is needed to determine output size for direct multi-step prediction.
        """
        if len(train_values) == 0:
            raise ValueError("train_values must not be empty")

        self.train_values = train_values

    def predict(self, horizon: int) -> list[float]:
        """
        Train an LSTM on the stored series and predict the next horizon steps.

        Parameters
        ----------
        horizon : int
            Forecast horizon.

        Returns
        -------
        list[float]
            Predicted values.
        """
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        train_arr = np.asarray(self.train_values, dtype=np.float32)

        # Fallback if series is too short to create windows.
        if len(train_arr) <= self.input_window + horizon:
            last_value = float(train_arr[-1])
            return [last_value] * horizon

        X, y = create_lstm_windows(
            series_values=self.train_values,
            input_window=self.input_window,
            forecast_horizon=horizon,
        )

        if len(X) == 0:
            last_value = float(train_arr[-1])
            return [last_value] * horizon

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        self.model = SimpleLSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        # Use last input window for forecasting.
        last_window = train_arr[-self.input_window:]
        last_window_tensor = (
            torch.tensor(last_window, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(self.device)
        )

        self.model.eval()
        with torch.no_grad():
            forecast = self.model(last_window_tensor).cpu().numpy().reshape(-1)

        if len(forecast) != horizon or not np.all(np.isfinite(forecast)):
            last_value = float(train_arr[-1])
            return [last_value] * horizon

        return forecast.astype(float).tolist()

    def get_params(self) -> dict[str, Any]:
        """
        Return model parameters for logging.
        """
        return {
            "input_window": self.input_window,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
        }