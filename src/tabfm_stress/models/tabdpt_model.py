"""Wrapper for TabDPT (Layer 6 AI)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabDPTModel(BaseModel):
    """TabDPT: Denoising Transformer with FAISS retrieval.

    Pretrained on real-world tables (not synthetic). Uses retrieval to
    find relevant context rows, then ICL for prediction.
    """

    name: str = "tabdpt"

    def __init__(self, device: str | None = None, **kwargs: Any) -> None:
        """Initialize the model wrapper.

        Args:
            device: Target device. None = auto-detect.
            **kwargs: Passed to TabDPTClassifier.
        """
        super().__init__(device=device)
        from tabdpt import TabDPTClassifier

        self._model = TabDPTClassifier(**kwargs)
        self._classes: np.ndarray | None = None

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns."""
        if self._classes is None:
            raise RuntimeError("classes_ not available before fit().")
        return self._classes

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context and build retrieval index (no weight updates)."""
        self._classes = np.array(sorted(np.unique(y)))
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(X)
