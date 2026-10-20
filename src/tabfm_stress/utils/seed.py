"""Shared utility functions for reproducibility."""

import logging
import os
import random

import numpy as np
import torch as th

logger = logging.getLogger(__name__)


def set_seed(seed: int | None = 42) -> None:
    """Set random seeds for all relevant libraries to ensure reproducibility.

    Controls randomness in:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch (CPU + CUDA + cuDNN)
    - OS-level hash seed (consistent dict ordering)

    Args:
        seed: Integer seed value. If None, no seeding is applied.

    Example:
        >>> set_seed(42)
    """
    if seed is None:
        logger.info("No seed provided, skipping reproducibility setup.")
        return

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    logger.info(f"Seed set to {seed} for reproducibility.")
