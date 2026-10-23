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
    """

    name: str = "kumo"

    def __init__(
        self,
        device: str | None = None,
        size: str = "medium",
        num_estimators: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialize the Kumo-Tabular wrapper."""
        super().__init__(device=device)
        import sdm
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = sdm.models.KumoTabular(
            task="classification",
            size=size,
            device=self._device,
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

        df = X.copy()
        if y is not None:
            df["_target"] = y

        stypes = sdm.infer_stypes(df, overrides={"_target": "categorical"})
        return sdm.TableTensor.from_pandas(df, stypes=stypes, device=self._device)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context as TableTensor (no weight updates)."""
        self._classes = np.array(sorted(np.unique(y)))

        table = self._to_table(X, y)
        self._x_context = table.drop_columns("_target")
        self._y_context = table["_target"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        classes = self._classes
        if classes is None:
            raise RuntimeError("fit() must be called before predict().")
        proba = self.predict_proba(X)
        return classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (single forward pass)."""
        import torch

        x_ctx = self._x_context
        y_ctx = self._y_context
        classes = self._classes
        if x_ctx is None or y_ctx is None or classes is None:
            raise RuntimeError("fit() must be called before predict_proba().")

        x_query = self._to_table(X)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=self._device.startswith("cuda"),
        ):
            probs = self._model(
                x_context=x_ctx,
                y_context=y_ctx,
                x_query=x_query,
                num_estimators=self._num_estimators,
            )

        proba_df = probs.to_pandas()
        return proba_df[[str(c) for c in classes]].values
