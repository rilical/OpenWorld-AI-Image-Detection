"""Train CLIP + residual fusion detector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from owaid.data import build_commfor_dataloaders
from owaid.inference import build_model_from_config
from owaid.training import run_training
from owaid.utils import (
    JsonlLogger,
    load_yaml,
    make_run_dir,
    merge_cli_overrides,
    save_resolved_config,
    set_seed,
    write_json,
    write_meta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the optional DIRE/fusion detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--opts", nargs="*")
    return parser.parse_args()


def _validate_config(cfg: dict) -> None:
    model_cfg = cfg.get("model", {})
    dire_cfg = model_cfg.get("dire", {})
    if model_cfg.get("type") != "clip_dire_fusion":
        return
    if not dire_cfg.get("enabled", False):
        return
    if not dire_cfg.get("cache_dir"):
        raise ValueError("DIRE training requires model.dire.cache_dir when the fusion path is enabled.")


def main() -> None:
    args = parse_args()
    cfg = merge_cli_overrides(load_yaml(args.config), args.opts)
    _validate_config(cfg)

    if args.validate_only:
        print("Config validation OK")
        return

    set_seed(int(cfg.get("seed", 0)), deterministic=bool(cfg.get("deterministic", True)))
    output_cfg = cfg.get("output", {})
    run_dir = make_run_dir(output_cfg.get("root", "outputs/runs"), output_cfg.get("run_name"))
    cfg.setdefault("output", {})["run_dir"] = run_dir
    save_resolved_config(run_dir, cfg)
    write_meta(run_dir, cfg)

    dataloaders = build_commfor_dataloaders(cfg, run_dir=run_dir)
    model = build_model_from_config(cfg).to(args.device)
    logger = JsonlLogger(str(Path(run_dir) / "logs" / "train.jsonl"))
    summary = run_training(
        cfg,
        model,
        dataloaders["train_fit"],
        dataloaders.get("val"),
        run_dir=run_dir,
        device=args.device,
        logger=logger,
    )
    write_json(str(Path(run_dir) / "train_summary.json"), summary)
    print(run_dir)


if __name__ == "__main__":
    main()
