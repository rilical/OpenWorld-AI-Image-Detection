"""Unified evaluation script for all datasets.

Usage:
    python scripts/evaluate.py --dataset commfor --config configs/eval_commfor.yaml --run outputs/runs/<run_id>
    python scripts/evaluate.py --dataset raid --config configs/eval_raid.yaml --run outputs/runs/<run_id>
    python scripts/evaluate.py --dataset vct2 --config configs/eval_vct2.yaml --run outputs/runs/<run_id>
    python scripts/evaluate.py --dataset aria --config configs/eval_aria.yaml --run outputs/runs/<run_id>

Replaces the separate eval_commfor.py, eval_raid.py, eval_vct2.py, eval_aria.py scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from scripts._common import evaluate_in_run, load_config_with_overrides

DATASET_MAP = {
    "commfor": ("commfor_small", "commfor"),
    "commfor_small": ("commfor_small", "commfor"),
    "raid": ("raid", "raid"),
    "vct2": ("vct2", "vct2"),
    "aria": ("aria", "aria"),
    "aiart": ("aiart", "aiart"),
    "cifake": ("cifake", "cifake"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on a benchmark dataset")
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_MAP.keys()),
        help="Dataset to evaluate on",
    )
    p.add_argument("--config", required=True, help="Eval config YAML path")
    p.add_argument("--run", required=True, help="Run directory path")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--evaluation-mode",
        choices=["all", "forced", "temperature", "threshold", "conformal"],
        default="all",
    )
    p.add_argument("--tau", type=float, default=0.9, help="Threshold for threshold abstention")
    p.add_argument("--opts", nargs="*")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name, artifact_name = DATASET_MAP[args.dataset]
    cfg = load_config_with_overrides(args.config, run_dir=args.run, overrides=args.opts)
    evaluate_in_run(
        cfg,
        args.run,
        dataset_name=dataset_name,
        artifact_name=artifact_name,
        device=args.device,
        save_predictions=True,
        evaluation_mode=args.evaluation_mode,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
