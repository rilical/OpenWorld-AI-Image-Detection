from __future__ import annotations

import numpy as np
import math

from owaid.calibration.conformal import build_split_conformal, prediction_set_from_probs


def test_quantile_uses_split_conformal_formula():
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
        ]
    )
    labels = np.array([0, 0, 1, 1, 1])
    alpha = 0.2
    report = build_split_conformal(probs, labels, alpha=alpha, method="split")
    n = len(labels)
    k = math.ceil((n + 1) * (1 - alpha))
    scores = 1 - probs[np.arange(n), labels]
    expected = np.partition(scores, k - 1)[k - 1]
    assert abs(report["qhat"] - expected) < 1e-8


def test_mondrian_includes_class_keys():
    probs = np.array([[0.6, 0.4], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
    labels = np.array([0, 1, 1, 0])
    report = build_split_conformal(probs, labels, alpha=0.1, method="mondrian")
    assert "class_qhat" in report
    assert set(report["class_qhat"].keys()) <= {"0", "1"}


def test_prediction_set_shape_and_values():
    probs = np.array([[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
    conformal = {"qhat": 0.6, "class_qhat": {"0": 0.5, "1": 0.5}, "method": "mondrian"}
    sets = prediction_set_from_probs(probs, conformal)
    assert len(sets) == len(probs)
    assert all(isinstance(s, list) for s in sets)
