"""Preprocessing for the TabFM-Stress benchmark.

TFMs handle encoding/scaling internally. The only step needed
is imputation (fit on context, transform context + test).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_context_test(
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Impute missing values using context statistics only.

    Returns:
        X_context, X_test (DataFrames with feature columns)
        y_context, y_test (1D numpy arrays)
    """
    X_context = context_df[feature_cols].copy()
    y_context = context_df[target_col].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values

    if X_context.isnull().any().any():
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        X_context = pd.DataFrame(
            imputer.fit_transform(X_context),
            columns=feature_cols,
            index=X_context.index,
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

    logger.info(f"Preprocessed: context={X_context.shape}, test={X_test.shape}")
    return X_context, X_test, y_context, y_test
