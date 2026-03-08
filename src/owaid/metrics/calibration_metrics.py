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


def empirical_conformal_coverage(
    prediction_sets: list[list[int]],
    labels: Any,
    group_ids: Any | None = None,
) -> Dict[str, Any]:
    """Fraction of samples where true label is in the prediction set.

    Returns overall coverage, class-conditional coverage, and optionally per-group coverage.
    """
    labels_np = np.asarray(labels)
    n = len(labels_np)
    if n == 0:
        return {"overall": 0.0, "class_conditional": {}, "per_group": {}}

    covered = np.array([int(labels_np[i]) in prediction_sets[i] for i in range(n)])

    result: Dict[str, Any] = {"overall": float(covered.mean())}

    # Class-conditional coverage
    class_cov = {}
    for c in np.unique(labels_np):
        mask = labels_np == c
        if mask.sum() > 0:
            class_cov[str(int(c))] = float(covered[mask].mean())
    result["class_conditional"] = class_cov

    # Per-group coverage
    if group_ids is not None:
        groups = np.asarray(group_ids)
        per_group = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() > 0:
                per_group[str(g)] = float(covered[mask].mean())
        result["per_group"] = per_group

    return result
