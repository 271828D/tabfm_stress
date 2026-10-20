"""Wrapper for TabPFN-3.5 (Prior Labs / SAP)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabPFNModel(BaseModel):
    """TabPFN-3.5: Prior-Fitted Network for tabular data.

    Approximates Bayesian inference over a prior on tabular datasets.
    Predicts in a single forward pass without weight updates.
    """

    name: str = "tabpfn"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model wrapper."""
        from tabpfn import TabPFNClassifier

        self._model = TabPFNClassifier(**kwargs)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        self._model.fit(x, y)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(x)
