"""Bootstrap confidence intervals for scalar metrics."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *args: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate a bootstrap confidence interval for ``metric_fn``."""
    arrays = [np.asarray(a) for a in args]
    if not arrays:
        raise ValueError("bootstrap_ci requires at least one array-like input")

    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All bootstrap inputs must have the same length")
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    rng = np.random.default_rng(seed)
    scores = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = [a[idx] for a in arrays]
        scores[i] = float(metric_fn(*resampled))

    alpha = 1.0 - confidence
    return {
        "mean": float(scores.mean()),
        "lower": float(np.percentile(scores, 100 * alpha / 2)),
        "upper": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "std": float(scores.std()),
    }
