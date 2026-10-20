"""Subject-aware splitting utilities."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_context_test(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    n_context: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split DataFrame into context and test sets at subject level.

    Randomly selects n_context subjects for context; all remaining
    subjects form the test set. No subject appears in both groups.

    Args:
        df: Input DataFrame.
        subject_col: Column with subject IDs.
        n_context: Number of subjects to use as context.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (context_df, test_df).

    Raises:
        ValueError: If subject_col is missing or n_context >= total subjects.
    """
    if subject_col not in df.columns:
        raise ValueError(f"Column '{subject_col}' not found in DataFrame.")

    subjects = df[subject_col].unique()
    total = len(subjects)

    if n_context >= total:
        raise ValueError(f"n_context ({n_context}) must be < total subjects ({total}).")

    rng = np.random.RandomState(random_state)
    chosen = rng.choice(subjects, size=n_context, replace=False)

    remaining = np.setdiff1d(subjects, chosen)

    context_df = df[df[subject_col].isin(chosen)].reset_index(drop=True)
    test_df = df[~df[subject_col].isin(chosen)].reset_index(drop=True)

    logger.info(
        f"Split: context={context_df[subject_col].nunique()} subjects "
        f"({len(context_df)} rows), "
        f"test={test_df[subject_col].nunique()} subjects "
        f"({len(test_df)} rows)"
    )

    return context_df, test_df, chosen.tolist(), remaining.tolist()
