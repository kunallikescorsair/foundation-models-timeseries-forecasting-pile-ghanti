"""
TimesFM foundation model wrapper.

This module provides a class-based wrapper around the official TimesFM
PyTorch inference API so it fits the shared project interface:
- fit(train_values)
- predict(horizon)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.models.base_model import BaseForecastModel

try:
    import torch
    import timesfm
    from huggingface_hub import snapshot_download

    TIMESFM_AVAILABLE = True
except ImportError:
    torch = None
    timesfm = None
    snapshot_download = None
    TIMESFM_AVAILABLE = False


class TimesFMModel(BaseForecastModel):
    """
    Wrapper for the official TimesFM PyTorch checkpoint.

    Notes
    -----
    - Uses a shared loaded model across all instances to avoid repeated loading.
    - Performs zero-shot forecasting.
    - Uses only the most recent `max_context` values from each training series.
    - Invalid TimesFM forecasts are treated as failed runs, not replaced with
      naive forecasts. This avoids misleading benchmark results.
    """

    name = "timesfm"

    _shared_model: ClassVar[Any | None] = None
    _shared_model_config: ClassVar[tuple | None] = None
    _shared_local_model_dir: ClassVar[str | None] = None

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 512,
        max_horizon: int = 256,
        normalize_inputs: bool = True,
        use_continuous_quantile_head: bool = True,
        force_flip_invariance: bool = True,
        infer_is_positive: bool = True,
        fix_quantile_crossing: bool = True,
        debug: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.normalize_inputs = normalize_inputs
        self.use_continuous_quantile_head = use_continuous_quantile_head
        self.force_flip_invariance = force_flip_invariance
        self.infer_is_positive = infer_is_positive
        self.fix_quantile_crossing = fix_quantile_crossing
        self.debug = debug

        self.train_values: list[float] | np.ndarray | None = None

    @classmethod
    def _resolve_local_model_dir(cls, model_name: str) -> str:
        """
        Resolve the TimesFM checkpoint to a local directory once.
        """
        if not TIMESFM_AVAILABLE:
            raise ImportError(
                "TimesFM is not installed. Install the official TimesFM package first."
            )

        if cls._shared_local_model_dir is not None:
            return cls._shared_local_model_dir

        candidate_path = Path(model_name)

        if candidate_path.exists():
            cls._shared_local_model_dir = str(candidate_path.resolve())
            return cls._shared_local_model_dir

        local_dir = snapshot_download(
            repo_id=model_name,
            repo_type="model",
        )

        cls._shared_local_model_dir = str(Path(local_dir).resolve())
        return cls._shared_local_model_dir

    @classmethod
    def _get_or_load_model(
        cls,
        model_name: str,
        max_context: int,
        max_horizon: int,
        normalize_inputs: bool,
        use_continuous_quantile_head: bool,
        force_flip_invariance: bool,
        infer_is_positive: bool,
        fix_quantile_crossing: bool,
    ) -> Any:
        """
        Load the shared TimesFM model once and reuse it.
        """
        if not TIMESFM_AVAILABLE:
            raise ImportError(
                "TimesFM is not installed. Install the official TimesFM package first."
            )

        current_config = (
            model_name,
            max_context,
            max_horizon,
            normalize_inputs,
            use_continuous_quantile_head,
            force_flip_invariance,
            infer_is_positive,
            fix_quantile_crossing,
        )

        if cls._shared_model is not None and cls._shared_model_config == current_config:
            return cls._shared_model

        local_model_dir = cls._resolve_local_model_dir(model_name)

        torch.set_float32_matmul_precision("high")

        model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
            model_id=local_model_dir,
            revision=None,
            cache_dir=None,
            force_download=False,
            local_files_only=True,
            token=None,
            config=None,
        )

        model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=normalize_inputs,
                use_continuous_quantile_head=use_continuous_quantile_head,
                force_flip_invariance=force_flip_invariance,
                infer_is_positive=infer_is_positive,
                fix_quantile_crossing=fix_quantile_crossing,
            )
        )

        cls._shared_model = model
        cls._shared_model_config = current_config

        return model

    def fit(self, train_values: list[float] | np.ndarray) -> None:
        """
        Store train values for zero-shot forecasting.
        """
        train_arr = np.asarray(train_values, dtype=float).reshape(-1)

        if train_arr.size == 0:
            raise ValueError("train_values must not be empty")

        self.train_values = train_arr

    def _prepare_train_array(self) -> np.ndarray:
        """
        Clean and trim training values before TimesFM inference.
        """
        if self.train_values is None:
            raise ValueError("Model must be fitted before prediction")

        train_arr = np.asarray(self.train_values, dtype=float).reshape(-1)

        if train_arr.size == 0:
            raise ValueError("No training values available")

        if train_arr.size > self.max_context:
            train_arr = train_arr[-self.max_context :]

        if not np.all(np.isfinite(train_arr)):
            mean_val = np.nanmean(train_arr)

            if not np.isfinite(mean_val):
                raise ValueError("Training series contains no finite values")

            train_arr = np.nan_to_num(
                train_arr,
                nan=float(mean_val),
                posinf=float(mean_val),
                neginf=float(mean_val),
            )

        if np.std(train_arr) == 0:
            raise ValueError("Training series is constant; TimesFM forecast is not meaningful")

        return train_arr

    def predict(self, horizon: int) -> list[float]:
        """
        Produce a zero-shot forecast using TimesFM.

        Parameters
        ----------
        horizon : int
            Forecast horizon.

        Returns
        -------
        list[float]
            Predicted values.

        Raises
        ------
        ValueError
            If TimesFM returns invalid, NaN, infinite, or wrong-length forecasts.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        effective_horizon = min(horizon, self.max_horizon)
        train_arr = self._prepare_train_array()

        model = self._get_or_load_model(
            model_name=self.model_name,
            max_context=self.max_context,
            max_horizon=self.max_horizon,
            normalize_inputs=self.normalize_inputs,
            use_continuous_quantile_head=self.use_continuous_quantile_head,
            force_flip_invariance=self.force_flip_invariance,
            infer_is_positive=self.infer_is_positive,
            fix_quantile_crossing=self.fix_quantile_crossing,
        )

        point_forecast, _ = model.forecast(
            horizon=effective_horizon,
            inputs=[train_arr],
        )

        forecast = np.asarray(point_forecast[0], dtype=float).reshape(-1)

        if self.debug:
            print("DEBUG TimesFM train shape:", train_arr.shape)
            print("DEBUG TimesFM train min:", np.min(train_arr))
            print("DEBUG TimesFM train max:", np.max(train_arr))
            print("DEBUG TimesFM train std:", np.std(train_arr))
            print("DEBUG TimesFM raw forecast:", forecast)

        if effective_horizon < horizon:
            if forecast.size == 0:
                raise ValueError("TimesFM produced empty forecast")

            pad_value = float(forecast[-1])
            padded = np.full(shape=horizon, fill_value=pad_value, dtype=float)
            padded[:effective_horizon] = forecast
            forecast = padded

        if forecast.size != horizon:
            raise ValueError(
                f"TimesFM produced wrong forecast length: "
                f"expected {horizon}, got {forecast.size}"
            )

        if not np.all(np.isfinite(forecast)):
            raise ValueError("TimesFM produced invalid forecast containing NaN or infinity")

        return forecast.tolist()

    def get_params(self) -> dict[str, Any]:
        """
        Return model parameters for logging.
        """
        return {
            "model_name": self.model_name,
            "max_context": self.max_context,
            "max_horizon": self.max_horizon,
            "normalize_inputs": self.normalize_inputs,
            "use_continuous_quantile_head": self.use_continuous_quantile_head,
            "force_flip_invariance": self.force_flip_invariance,
            "infer_is_positive": self.infer_is_positive,
            "fix_quantile_crossing": self.fix_quantile_crossing,
        }