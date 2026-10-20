"""Wrapper for TabFM (Google Research)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabFMModel(BaseModel):
    """TabFM: Hybrid-Attention ICL transformer (24 layers, 400M params).

    Combines TabPFN-style alternating row/column attention with
    TabICL-style row compression. Trained on synthetic SCMs.
    Highest TabArena accuracy among open models.
    """

    name: str = "tabfm"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model wrapper."""
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

        self._model = tabfm_v1_0_0.load(model_type="classification")
        self._clf = TabFMClassifier(model=self._model, **kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        self._clf.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._clf.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self._clf.predict_proba(X)
