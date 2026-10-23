"""Wrapper for Kumo-Tabular (NVIDIA)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)


class KumoModel(BaseModel):
    """Kumo-Tabular: Pure ICL transformer (28M–215M params).

    Pretrained on synthetic SCMs. #1 on TabArena (Elo 1950).
    Supports classification and regression.
    """

    name: str = "kumo"

    def __init__(
        self,
        device: str | None = None,
        size: str = "large",
        num_estimators: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialize the Kumo-Tabular wrapper.

        Args:
            device: Target device. None = auto-detect.
            size: Model size: "small" (28M), "medium" (~100M), "large" (215M).
            num_estimators: Number of ICL ensembles (averaged).
            **kwargs: Additional parameters passed to KumoTabular.
        """
        super().__init__(device=device)
        import sdm

        resolved_device = device or (
            "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
        self._model = sdm.models.KumoTabular(
            task="classification",
            size=size,
            device=resolved_device,
            **kwargs,
        )
        self._num_estimators = num_estimators
        self._classes: np.ndarray | None = None
        self._x_context: Any = None
        self._y_context: Any = None

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns."""
        if self._classes is None:
            raise RuntimeError("classes_ not available before fit().")
        return self._classes

    def _to_table(self, X: pd.DataFrame, y: np.ndarray | None = None) -> Any:
        """Convert DataFrame (+ optional target) to sdm.TableTensor."""
        import sdm

        resolved_device = (
            self._model.device if hasattr(self._model, "device") else "cuda"
        )
        df = X.copy()
        if y is not None:
            df["_target"] = y

        stypes = sdm.infer_stypes(df, overrides={"_target": "categorical"})
        return sdm.TableTensor.from_pandas(df, stypes=stypes, device=resolved_device)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Cache context rows for ICL via KV-cache (no weight updates)."""
        self._classes = np.array(sorted(np.unique(y)))

        table = self._to_table(X, y)
        self._x_context = table.drop_columns("_target")
        self._y_context = table["_target"]

        # Use fit() to cache context encoding (faster subsequent predict calls)
        self._model.fit(
            x=self._x_context,
            y=self._y_context,
            num_estimators=self._num_estimators,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        assert self._classes is not None
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using cached context."""
        import torch

        table = self._to_table(X)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=str(self._model.device).startswith("cuda"),
        ):
            probs = self._model.predict(table, num_estimators=self._num_estimators)

        # Convert TableTensor → numpy array
        proba_df = probs.to_pandas()
        # Align columns to self._classes order
        return proba_df.values

    def clear(self) -> None:
        """Free the KV-cache (call between seeds to release VRAM)."""
        self._model.clear()
        self._x_context = None
        self._y_context = None
