"""Wrapper for Causilo (Nums AI)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)


class CausiloModel(BaseModel):
    """Causilo: Latent-Token ICL (4 tokens per row).

    Compresses each row into 4 latent tokens, then performs ICL.
    Tops TabArena among single models (Sep 2026).
    """

    name: str = "causilo"

    def __init__(self, device: str | None = None, **kwargs: Any) -> None:
        """Initialize the model wrapper.

        Args:
            device: Target device. None = auto-detect.
            **kwargs: Passed to CausiloClassifier.
        """
        super().__init__(device=device)
        from causilo import CausiloClassifier

        self._model = CausiloClassifier(**kwargs)
        self._classes: np.ndarray | None = None

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns."""
        if self._classes is None:
            raise RuntimeError("classes_ not available before fit().")
        return self._classes

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Store context rows for ICL (no weight updates)."""
        self._classes = np.array(sorted(y.unique()))
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(X)
