"""Debug the data pipeline without running any model."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only

import numpy as np

from tabfm_stress.data.loader import load_data
from tabfm_stress.data.preprocess import preprocess_context_test
from tabfm_stress.data.splitting import split_context_test
from tabfm_stress.evaluation.artifacts import (
    save_predictions,
    save_split_data,
)

# --- Load & split ---
df = load_data(
    "data/openface_features_with_metadata_stressID_resized_ExtraFeatures.csv"
)

target_col = "label"
subject_col = "subject_id"
feature_cols = [
    c
    for c in df.select_dtypes(include="number").columns
    if c not in (target_col, subject_col)
]

print(f"Feature cols ({len(feature_cols)}): {feature_cols}")

ctx_df, test_df, ctx_subjects, test_subjects = split_context_test(
    df, subject_col=subject_col, n_context=5, random_state=0
)
print(f"Context: {ctx_df.shape}, Test: {test_df.shape}")

# --- Preprocess ---
X_ctx, X_test, y_ctx, y_test = preprocess_context_test(
    ctx_df, test_df, feature_cols, target_col
)
print(f"X_ctx: {X_ctx.shape}, X_test: {X_test.shape}")

# --- Save split data ---
save_split_data(
    output_dir="results/debug/",
    seed=0,
    n_context=5,
    ctx_df=ctx_df,
    test_df=test_df,
    X_context=X_ctx,
    X_test=X_test,
    y_context=y_ctx,
    y_test=y_test,
)

# --- Mock predictions (random) ---
n = len(y_test)
y_pred = np.random.choice([0, 1], size=n)
y_proba = np.column_stack([1 - np.random.rand(n), np.random.rand(n)])

save_predictions(
    output_dir="results/debug/",
    model_name="mock",
    seed=0,
    n_context=5,
    y_test=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    X_test=X_test,
    paths=test_df["path"].values,
)

print("\nDone. Check results/debug/")
