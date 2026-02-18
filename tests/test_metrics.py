from __future__ import annotations

import numpy as np

from owaid.metrics.classification import auroc, tpr_at_fpr
from owaid.metrics.calibration_metrics import ece
from owaid.metrics.selective import risk_coverage, selective_accuracy


def test_auroc_with_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert abs(auroc(y_true, y_score) - 1.0) < 1e-6


def test_tpr_at_fpr_one_percent():
    y_true = [0, 0, 1, 1]
    y_score = [0.05, 0.1, 0.9, 0.95]
    value = tpr_at_fpr(y_true, y_score, fpr=0.01)
    assert value <= 1.0


def test_ece_sanity_low_for_perfect_calibration():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    ece_value, _ = ece(probs, labels, n_bins=4)
    assert ece_value < 0.15


def test_selective_metrics_smoke():
    conf = np.array([0.99, 0.95, 0.2, 0.1])
    corr = np.array([1, 0, 1, 0], dtype=bool)
    rc = risk_coverage(conf, corr)
    assert set(rc.keys()) == {"coverage", "risk", "aurc"}
    assert rc["aurc"] >= 0.0


def test_selective_accuracy_zero_when_no_answered():
    conf = np.array([0.1])
    corr = np.array([1], dtype=bool)
    assert selective_accuracy(corr, np.array([False])) == 0.0
