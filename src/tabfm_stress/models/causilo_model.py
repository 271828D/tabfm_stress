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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model wrapper."""
        from causilo import CausiloClassifier

        self._model = CausiloClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(X)
