"""Wrapper for TabFM (Google Research)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabfm_stress.models.base import BaseModel

logger = logging.getLogger(__name__)

# ── Module-level weight cache (paid once per process) ──────────────────────
_tabfm_weights: Any = None


def _load_tabfm_weights(device: str | None = None) -> Any:
    """Load TabFM weights once; return cached reference on subsequent calls."""
    global _tabfm_weights
    if _tabfm_weights is None:
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

        _tabfm_weights = tabfm_v1_0_0.load(model_type="classification")
        logger.info("TabFM weights loaded.")
    if device is not None:
        _tabfm_weights = _tabfm_weights.to(device)
    return _tabfm_weights


class TabFMModel(BaseModel):
    """TabFM: Hybrid-Attention ICL transformer (24 layers, 400M params).

    Combines TabPFN-style alternating row/column attention with
    TabICL-style row compression. Trained on synthetic SCMs.
    Highest TabArena accuracy among open models.
    """

    name: str = "tabfm"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Initialize the model wrapper.

        Args:
            device: Target device (e.g. "cuda", "cpu"). None = auto-detect.
            batch_size: Number of test rows per forward pass during predict.
            **kwargs: Additional parameters passed to TabFMClassifier.
        """
        super().__init__(device=device)
        from tabfm import TabFMClassifier

        self._model = _load_tabfm_weights(device)
        self._clf = TabFMClassifier(model=self._model, **kwargs)
        self._batch_size = batch_size
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
        self._clf.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels with manual batching to control VRAM."""
        preds = []
        for i in range(0, len(X), self._batch_size):
            batch = X.iloc[i : i + self._batch_size]
            preds.append(self._clf.predict(batch))
        return np.concatenate(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities with manual batching to control VRAM."""
        probas = []
        for i in range(0, len(X), self._batch_size):
            batch = X.iloc[i : i + self._batch_size]
            probas.append(self._clf.predict_proba(batch))
        return np.concatenate(probas, axis=0)
