"""Benchmark runner: iterates over seeds × context sizes × models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from tabfm_stress.data.loader import load_data
from tabfm_stress.data.preprocess import preprocess_context_test
from tabfm_stress.data.splitting import split_context_test
from tabfm_stress.evaluation.artifacts import save_predictions, save_split_data
from tabfm_stress.evaluation.metrics import compute_metrics
from tabfm_stress.models.zoo import MODELS, get_model
from tabfm_stress.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_benchmark(
    data_path: str,
    model_names: list[str] | None = None,
    seeds: list[int] | None = None,
    n_context_list: list[int] | None = None,
    feature_cols: list[str] | None = None,
    target_col: str = "label",
    subject_col: str = "subject_id",
    output_dir: str = "results/",
    device: str = "cpu",
) -> pd.DataFrame:
    """Run the full zero-shot benchmark.

    For each seed and context size, splits the data, then evaluates
    each model on the resulting context/test pair.

    Args:
        data_path: Path to the dataset (parquet or CSV).
        model_names: Subset of models to run (None = all).
        seeds: List of random seeds (default: [0, 1, 2]).
        n_context_list: Number of context subjects per run.
        feature_cols: Feature column names (None = auto-detect numeric).
        target_col: Label column name.
        subject_col: Subject ID column name.
        output_dir: Where to save the results.
        device: Compute device ("cpu" or "cuda").

    Returns:
        DataFrame with one row per (seed, context_size, model).
    """
    run_dir = Path(output_dir) / f"run_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning(
                "device='cuda' requested but no CUDA device found — falling back to CPU"
            )
            device = "cpu"
        else:
            logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"Device: {device}")

    if model_names is None:
        model_names = list(MODELS.keys())
    if seeds is None:
        seeds = [0, 1, 2]
    if n_context_list is None:
        n_context_list = [5, 13, 26, 40, 47]

    df = load_data(data_path)

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in (target_col, subject_col)
        ]

    # Issue 7: log feature info once, outside the loop
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    rows: list[dict[str, Any]] = []

    total_trials = len(seeds) * len(n_context_list) * len(model_names)
    pbar = tqdm(total=total_trials, desc="Benchmark", unit="trial")

    for seed in seeds:
        for n_ctx in n_context_list:
            ctx_df, test_df, ctx_subjects, test_subjects = split_context_test(
                df,
                subject_col=subject_col,
                n_context=n_ctx,
                random_state=seed,
            )
            X_ctx, X_test, y_ctx, y_test = preprocess_context_test(
                ctx_df, test_df, feature_cols, target_col
            )

            logger.info(f"X_ctx: {X_ctx.shape}, X_test: {X_test.shape}")

            save_split_data(
                output_dir=run_dir,
                seed=seed,
                n_context=n_ctx,
                ctx_df=ctx_df,
                test_df=test_df,
                X_context=X_ctx,
                X_test=X_test,
                y_context=y_ctx,
                y_test=y_test,
            )

            for name in model_names:
                logger.info(f"Running {name} | seed={seed} | n_ctx={n_ctx}")
                set_seed(seed)
                try:
                    # Issue 5: forward device to model construction
                    model = get_model(name, device=device)
                    model.fit(X_ctx, y_ctx)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)
                    metrics = compute_metrics(y_test, y_pred, y_proba)
                except Exception:
                    logger.exception(
                        f"Model '{name}' failed | seed={seed} | n_ctx={n_ctx}"
                    )
                    pbar.update(1)  # Issue 3
                    continue

                rows.append(
                    {
                        "seed": seed,
                        "n_context_subjects": n_ctx,
                        "model": name,
                        **metrics,
                    }
                )

                # Issue 4: save_predictions guarded inside its own try
                try:
                    paths = (
                        test_df["path"].values if "path" in test_df.columns else None
                    )
                    save_predictions(
                        output_dir=run_dir,
                        model_name=name,
                        seed=seed,
                        n_context=n_ctx,
                        y_test=y_test,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        X_test=X_test,
                        paths=paths,
                    )
                except Exception:
                    logger.exception(f"Failed to save predictions for '{name}'")

                pbar.update(1)  # Issue 3

    pbar.close()

    results = pd.DataFrame(rows)
    out_path = run_dir / "benchmark.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}: {len(results)} rows")

    return results
