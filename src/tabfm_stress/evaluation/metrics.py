"""Centralized evaluation metrics for the TabFM-Stress benchmark."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calculate evaluation metrics for binary classification.

    Args:
        y_true: Actual labels (0 or 1), shape [n_samples].
        y_pred: Predicted labels (0 or 1), shape [n_samples].
        y_proba: Predicted probabilities, shape [n_samples, 2].
            Uses column 1 (positive class) for ROC-AUC.

    Returns:
        Dictionary with metric name → value.
    """
    # Extract positive-class probability for ROC-AUC
    y_prob_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob_pos),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    logger.info(
        f"Metrics computed. F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}"
    )
    return metrics
