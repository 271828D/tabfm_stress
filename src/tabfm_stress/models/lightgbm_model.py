"""LightGBM classifier wrapper for the TabFM-Stress benchmark."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from tabfm_stress.models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM gradient-boosted tree classifier.

    Wraps LGBMClassifier to conform to the uniform BaseModel interface.
    """

    name: str = "lightgbm"

    def __init__(
        self,
        device: str | None = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs: int | float | str,
    ) -> None:
        """Initialize the LightGBM wrapper.

        Args:
            device: Compute device ("cpu", "cuda", or "gpu"). None = cpu.
            n_estimators: Number of boosting rounds.
            learning_rate: Shrinkage factor.
            num_leaves: Maximum leaves per tree.
            random_state: Seed for reproducibility.
            **kwargs: Additional LGBMClassifier parameters.
        """
        super().__init__(device=device)
        lgb_device = "gpu" if device == "cuda" else (device or "cpu")
        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            device=lgb_device,
            verbose=-1,
            **kwargs,
        )
        self._fitted: bool = False

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns.

        Returns:
            Array of class labels.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("LightGBMModel must be fitted before accessing classes_")
        return self._model.classes_

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train the LightGBM classifier.

        Args:
            X: Training features (shape [n_samples, n_features]).
            y: Training labels (shape [n_samples,]).
        """
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features (shape [n_samples, n_features]).

        Returns:
            Predicted labels (shape [n_samples,]).
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features (shape [n_samples, n_features]).

        Returns:
            Probability matrix (shape [n_samples, n_classes]).
        """
        return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk.

        Args:
            path: Destination file path (.pkl).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
