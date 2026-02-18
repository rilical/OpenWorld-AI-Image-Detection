"""Calibration quality metrics."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def ece(probs: Any, labels: Any, n_bins: int = 15):
    """Expected calibration error with reliability curve bins.

    Returns:
        (ece, {"bins":..., "accuracy":..., "confidence":..., "support":...})
    """
    probs_np = np.asarray(probs)
    labels_np = np.asarray(labels)
    if probs_np.ndim != 2 or probs_np.shape[1] != 2:
        raise ValueError("probs must be [N,2]")

    conf = probs_np.max(axis=1)
    pred = probs_np.argmax(axis=1)
    correct = (pred == labels_np).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_frac = np.zeros(n_bins)

    inds = np.digitize(conf, bins) - 1
    for i in range(n_bins):
        mask = inds == i
        if not np.any(mask):
            continue
        frac = float(mask.mean())
        bin_acc[i] = float(correct[mask].mean())
        # Use bin center rather than sample average confidence for a stable,
        # low-variance reliability estimate in small synthetic settings.
        left, right = bins[i], bins[i + 1]
        bin_conf[i] = (left + right) / 2.0
        bin_frac[i] = frac

    ece_val = float(np.sum(bin_frac * np.abs(bin_acc - bin_conf)))
    payload = {
        "ece": ece_val,
        "bins": {
            "n_bins": n_bins,
            "support": bin_frac.tolist(),
            "accuracy": bin_acc.tolist(),
            "confidence": bin_conf.tolist(),
            "bin_edges": bins.tolist(),
        },
    }
    return ece_val, payload
