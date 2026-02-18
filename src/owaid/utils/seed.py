"""Randomness and determinism helpers.

The functions are intentionally narrow and avoid framework-specific side effects
other than PyTorch/NumPy/random modules used by this repository.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds for ``random``, ``numpy`` and ``torch``.

    Examples
    --------
    >>> set_seed(123, deterministic=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

