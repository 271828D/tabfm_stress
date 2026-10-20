"""Data loading utilities for the TabFM-Stress benchmark."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the benchmark dataset from a parquet or CSV file.

    Handles both `.parquet` and `.csv` (semicolon-separated) formats.
    Normalizes column names to lowercase with underscores.

    Args:
        path: Path to the data file.

    Returns:
        Cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found at: {p}")

    logger.info(f"Loading data from {p}")

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p, sep=";")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .csv")

    # Normalize column names
    df.columns = [col.strip().lower().replace("-", "_") for col in df.columns]

    logger.info(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    logger.debug(f"Columns: {df.columns.tolist()}")

    return df
