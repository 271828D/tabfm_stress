"""Save per-run artifacts for post-hoc analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_split_data(
    output_dir: str | Path,
    seed: int,
    n_context: int,
    ctx_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_context: pd.DataFrame,
    X_test: pd.DataFrame,
    y_context: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Save the full split data (context + test) for one (seed, n_context) pair."""
    split_dir = Path(output_dir) / "splits" / f"seed_{seed}_ctx_{n_context}"
    split_dir.mkdir(parents=True, exist_ok=True)

    ctx_out = pd.DataFrame(
        {
            "split": "context",
            "path": ctx_df["path"].values,
            "y_true": y_context,
        }
    )
    ctx_out = pd.concat([ctx_out, X_context.reset_index(drop=True)], axis=1)

    test_out = pd.DataFrame(
        {
            "split": "test",
            "path": test_df["path"].values,
            "y_true": y_test,
        }
    )
    test_out = pd.concat([test_out, X_test.reset_index(drop=True)], axis=1)

    full = pd.concat([ctx_out, test_out], ignore_index=True)
    full.to_csv(split_dir / "split_data.csv", index=False)
    logger.debug(f"Saved split data: {split_dir}")


def save_predictions(
    output_dir: str | Path,
    model_name: str,
    seed: int,
    n_context: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    paths: np.ndarray | None = None,
    X_test: pd.DataFrame | None = None,
) -> None:
    """Save predictions for one model run.

    Args:
        output_dir: Base output directory.
        model_name: Model identifier.
        seed: Random seed.
        n_context: Number of context subjects.
        y_test: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities, shape [n_samples, 2].
        paths: Optional file paths for traceability.
        X_test: Optional feature DataFrame to include in output.
    """
    run_dir = Path(output_dir) / model_name / f"seed_{seed}_ctx_{n_context}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba_pos": y_proba[:, 1],
        }
    )
    if paths is not None:
        pred_df.insert(0, "path", paths)
    if X_test is not None:
        pred_df = pd.concat([pred_df, X_test.reset_index(drop=True)], axis=1)

    pred_df.to_csv(run_dir / "predictions.csv", index=False, sep=";")

    logger.debug(f"Saved predictions: {run_dir}")
