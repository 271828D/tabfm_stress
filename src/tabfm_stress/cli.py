"""Command-line interface for the TabFM-Stress benchmark."""

from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only


def main() -> None:
    """Entry point for the tabfm-stress CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="tabfm-stress",
        description="Zero-shot tabular foundation model benchmark",
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- run subcommand ---
    run = sub.add_parser("run", help="Run the benchmark")
    run.add_argument(
        "--data",
        type=str,
        default="data/openface_features_with_metadata_stressID_resized_ExtraFeatures.csv",
        help="Path to dataset (.parquet or .csv)",
    )
    run.add_argument(
        "--models",
        type=str,
        default="all",
        help='Comma-separated model names, or "all" (default: all)',
    )
    run.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    run.add_argument(
        "--n-context",
        type=int,
        nargs="+",
        default=[5, 13, 26, 40, 47],
        help="Number of context subjects per run (default: 5 13 26 40 47)",
    )
    run.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated feature columns (default: auto-detect)",
    )
    run.add_argument(
        "--target-col",
        type=str,
        default="label",
        help="Target column name (default: label)",
    )
    run.add_argument(
        "--subject-col",
        type=str,
        default="subject_id",
        help="Subject ID column name (default: subject_id)",
    )
    run.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory (default: results/)",
    )

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    if args.cmd == "run":
        from tabfm_stress.evaluation.runner import run_benchmark

        model_names = args.models.split(",") if args.models != "all" else None
        feature_cols = args.feature_cols.split(",") if args.feature_cols else None

        run_benchmark(
            data_path=args.data,
            model_names=model_names,
            seeds=args.seeds,
            n_context_list=args.n_context,
            feature_cols=feature_cols,
            target_col=args.target_col,
            subject_col=args.subject_col,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
