"""
Uni2TS (Moirai) foundation model wrapper.

Zero-shot forecasting using pretrained Moirai model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecastModel

try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    UNI2TS_AVAILABLE = True
except ImportError:
    UNI2TS_AVAILABLE = False


class Uni2TSModel(BaseForecastModel):
    name = "uni2ts"

    def __init__(
        self,
        model_name: str = "Salesforce/moirai-1.1-R-small",
        context_length: int = 200,
        num_samples: int = 20,
        normalized_frequency: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.context_length = context_length
        self.num_samples = num_samples
        self.normalized_frequency = normalized_frequency

        self.train_values: list[float] | None = None

    def fit(self, train_values: list[float]) -> None:
        if len(train_values) == 0:
            raise ValueError("train_values must not be empty")

        self.train_values = train_values

    def _get_pandas_freq(self) -> str:
        mapping = {
            "weekly": "W",
            "daily": "D",
            "monthly": "MS",
            "quarterly": "QS",
            "yearly": "YS",
            "hourly": "H",
            "half_hourly": "30min",
            "10_minutes": "10min",
        }

        if self.normalized_frequency not in mapping:
            raise ValueError(f"Unsupported frequency: {self.normalized_frequency}")

        return mapping[self.normalized_frequency]

    def predict(self, horizon: int) -> list[float]:
        if not UNI2TS_AVAILABLE:
            raise ImportError("uni2ts is not installed")

        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")

        train_arr = np.asarray(self.train_values, dtype=float)

        # limit context
        if len(train_arr) > self.context_length:
            train_arr = train_arr[-self.context_length:]

        train_arr = np.nan_to_num(train_arr, nan=float(np.nanmean(train_arr)))

        # build dataframe
        freq = self._get_pandas_freq()

        df = pd.DataFrame(
            {"value": train_arr},
            index=pd.date_range(
                start="2000-01-01",
                periods=len(train_arr),
                freq=freq,
            ),
        )

        ds = PandasDataset({"value": df["value"]})

        train, test_template = split(ds, offset=-horizon)
        test_data = test_template.generate_instances(
            prediction_length=horizon,
            windows=1,
            distance=horizon,
        )

        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self.model_name),
            prediction_length=horizon,
            context_length=self.context_length,
            patch_size="auto",
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

        predictor = model.create_predictor(batch_size=8)
        forecasts = list(predictor.predict(test_data.input))

        forecast = forecasts[0].mean

        forecast = np.asarray(forecast, dtype=float)

        if len(forecast) != horizon or not np.all(np.isfinite(forecast)):
            last_val = float(train_arr[-1])
            return [last_val] * horizon

        return forecast.tolist()

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "context_length": self.context_length,
            "num_samples": self.num_samples,
        }