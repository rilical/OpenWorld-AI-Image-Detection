"""Generate quick artifacts from evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metrics", nargs="*", default=["auroc", "tpr_at_1pct_fpr", "abstain_rate", "selective_accuracy"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.runs)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = [p for p in run_root.iterdir() if p.is_dir()]
    all_run_metrics = {}
    for run in runs:
        for metric_name in ["commfor", "vct2", "raid", "aria"]:
            mpath = run / "eval" / metric_name / "metrics.json"
            if mpath.exists():
                with mpath.open("r", encoding="utf-8") as f:
                    all_run_metrics.setdefault(str(run), {})[metric_name] = json.load(f)

    for metric in args.metrics:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        x = list(range(len(all_run_metrics)))
        values = []
        labels = []
        for name, metric_map in all_run_metrics.items():
            # prefer VCT2, fallback to commfor
            record = metric_map.get("vct2") or metric_map.get("commfor") or {}
            if metric in record:
                values.append(record[metric])
                labels.append(Path(name).name)
        if not values:
            continue
        ax.bar(labels, values)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out_root / f"{metric}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
