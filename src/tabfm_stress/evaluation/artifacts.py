"""Save per-run artifacts for post-hoc analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_split_info(
    output_dir: str | Path,
    seed: int,
    n_context: int,
    context_subjects: list[str],
    test_subjects: list[str],
) -> None:
    """Save which subjects were assigned to context vs test.

    Saved once per (seed, n_context) pair — shared across all models.

    Args:
        output_dir: Base output directory.
        seed: Random seed used for the split.
        n_context: Number of context subjects.
        context_subjects: Subject IDs used as context.
        test_subjects: Subject IDs used for evaluation.
    """
    split_dir = Path(output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "seed": seed,
        "n_context_subjects": n_context,
        "context_subjects": context_subjects,
        "test_subjects": test_subjects,
    }
    path = split_dir / f"seed_{seed}_ctx_{n_context}.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)

    logger.debug(f"Saved split info: {path}")


def save_predictions(
    output_dir: str | Path,
    model_name: str,
    seed: int,
    n_context: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
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
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    logger.debug(f"Saved predictions: {run_dir}")
