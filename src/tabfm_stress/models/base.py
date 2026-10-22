"""Abstract base class for all tabular foundation model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Uniform interface for all tabular foundation model wrappers.

    Every model wrapper must implement fit, predict, and predict_proba
    so the benchmark runner can treat them identically.
    """

    name: str = "base"

    def __init__(self, device: str | None = None) -> None:
        """Initialize the model wrapper.

        Args:
            device: Compute device ("cpu", "cuda"). None = auto-detect.
        """
        self.device = device

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns.

        Must be available after fit() is called.
        """
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context for in-context learning (no weight updates).

        Args:
            X: Context features (shape [n_samples, n_features]).
            y: Context labels (shape [n_samples,]).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the test set.

        Args:
            X: Test features (shape [n_samples, n_features]).

        Returns:
            Predicted labels (shape [n_samples,]).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the test set.

        Column order corresponds to self.classes_.

        Args:
            X: Test features (shape [n_samples, n_features]).

        Returns:
            Probability matrix (shape [n_samples, n_classes]).
        """
        ...
