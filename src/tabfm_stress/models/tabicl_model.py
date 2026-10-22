"""Wrapper for TabICLv2 (Inria)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabICLModel(BaseModel):
    """TabICLv2: Scalable In-Context Learning for tabular data.

    Two-stage pipeline: column-wise embedding then row-wise ICL attention.
    Scales to 100K+ rows. BSD-3 licensed (commercially usable).
    """

    name: str = "tabicl"

    def __init__(self, device: str | None = None, **kwargs: Any) -> None:
        """Initialize the model wrapper.

        Args:
            device: Target device. None = auto-detect.
            **kwargs: Passed to TabICLClassifier.
        """
        super().__init__(device=device)
        from tabicl import TabICLClassifier

        self._model = TabICLClassifier(**kwargs)
        self._classes: np.ndarray | None = None

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns."""
        if self._classes is None:
            raise RuntimeError("classes_ not available before fit().")
        return self._classes

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        self._classes = np.array(sorted(np.unique(y)))
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(X)
