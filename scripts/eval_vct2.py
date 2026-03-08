"""Evaluate model on VCT2."""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
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
    cfg = load_config_with_overrides(args.config, run_dir=args.run, overrides=args.opts)
    evaluate_in_run(
        cfg=cfg,
        run_dir=args.run,
        dataset_name="vct2",
        artifact_name="vct2",
        device=args.device,
        save_predictions=True,
        evaluation_mode=args.evaluation_mode,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
