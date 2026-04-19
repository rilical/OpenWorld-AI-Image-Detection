"""Ensemble-disagreement abstention over baseline + DIRE + SGF predictions.

Reads the forced-mode predictions parquet from each run, aligns on sample_id,
and computes three abstention policies:

  - unanimous  : predict only if all three models agree; otherwise abstain
  - majority   : predict the majority vote; no abstention
  - prob-std   : abstain when std of prob_ai across models > tau (default 0.15)

Reports accuracy, coverage, abstain rate and confusion-matrix stats on
CommunityForensics and RAID. Writes JSON to reports/ensemble_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {
    "baseline": "outputs/runs/20260416_175137_a827a8",
    "dire":     "outputs/runs/20260417_005521_91ec33",
    "sgf":      "outputs/runs/20260416_175137_8eb073",
}

DATASETS = ("commfor", "raid")
STD_TAU = 0.15


def _load(ds: str):
    frames = {}
    for tag, run in RUNS.items():
        path = Path(run) / "eval" / ds / "predictions_forced.parquet"
        df = pd.read_parquet(path)[["sample_id", "label", "pred", "prob_ai"]]
        df = df.rename(columns={"pred": f"pred_{tag}", "prob_ai": f"prob_{tag}"})
        if "merged" not in locals():
            merged = df
        else:
            merged = merged.merge(df, on=["sample_id", "label"], how="inner")
        frames[tag] = df
    return merged


def _policy_metrics(y: np.ndarray, pred: np.ndarray, abstain: np.ndarray):
    kept = ~abstain
    n = len(y)
    cov = float(kept.mean())
    ab = float(abstain.mean())
    if kept.sum() == 0:
        return {"coverage": cov, "abstain_rate": ab, "selective_accuracy": None,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0, "tpr": None, "fpr": None}
    yk, pk = y[kept], pred[kept]
    tp = int(((pk == 1) & (yk == 1)).sum())
    tn = int(((pk == 0) & (yk == 0)).sum())
    fp = int(((pk == 1) & (yk == 0)).sum())
    fn = int(((pk == 0) & (yk == 1)).sum())
    acc = (tp + tn) / len(yk)
    pos = (yk == 1).sum()
    neg = (yk == 0).sum()
    tpr = tp / pos if pos > 0 else None
    fpr = fp / neg if neg > 0 else None
    return {"coverage": cov, "abstain_rate": ab, "selective_accuracy": float(acc),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tpr": tpr, "fpr": fpr, "n_kept": int(kept.sum())}


def main() -> None:
    results = {}
    for ds in DATASETS:
        m = _load(ds)
        y = m["label"].to_numpy()
        preds = np.stack([m[f"pred_{t}"].to_numpy() for t in RUNS], axis=1)
        probs = np.stack([m[f"prob_{t}"].to_numpy() for t in RUNS], axis=1)
        majority = (preds.sum(axis=1) >= 2).astype(int)
        unanimous_fake = (preds.sum(axis=1) == 3)
        unanimous_real = (preds.sum(axis=1) == 0)
        unanimous = unanimous_fake | unanimous_real
        prob_std = probs.std(axis=1)

        policies = {
            "majority_no_abstain": _policy_metrics(y, majority, np.zeros_like(y, dtype=bool)),
            "unanimous_abstain":   _policy_metrics(y, majority, ~unanimous),
            f"prob_std_tau_{STD_TAU}": _policy_metrics(y, majority, prob_std > STD_TAU),
        }
        # Individual models as reference
        for t in RUNS:
            policies[f"single_{t}"] = _policy_metrics(
                y, m[f"pred_{t}"].to_numpy(), np.zeros_like(y, dtype=bool)
            )
        results[ds] = {"n_samples": int(len(y)), **policies}

    out = Path("reports/ensemble_metrics.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
