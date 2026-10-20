"""Abstract base class for all tabular foundation model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Uniform interface for all tabular foundation models.

    Every model wrapper must implement fit, predict, and predict_proba
    so the benchmark runner can treat them identically.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store context for in-context learning (no weight updates).

        Args:
            X: Context features (2D array, shape [n_samples, n_features]).
            y: Context labels (1D array, shape [n_samples]).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the test set.

        Args:
            X: Test features (2D array, shape [n_samples, n_features]).

        Returns:
            Predicted labels (1D array, shape [n_samples]).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the test set.

        Args:
            X: Test features (2D array, shape [n_samples, n_features]).

        Returns:
            Probability matrix (2D array, shape [n_samples, n_classes]).
        """
        ...
