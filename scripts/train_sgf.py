"""Train SGF-Net: Spectral-Gated Forensic Fusion Network."""

from __future__ import annotations

import os
os.environ.setdefault(
    "HF_HOME",
    "/ocean/projects/cis250202p/gyar/personal/dl/OpenWorld-AI-Image-Detection/.cache/huggingface",
)

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from owaid.data import build_train_dataloaders
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
    parser = argparse.ArgumentParser(description="Train SGF-Net detector")
    parser.add_argument("--config", default="configs/sgf_net.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--opts", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_cli_overrides(load_yaml(args.config), args.opts)

    if cfg.get("model", {}).get("type") != "sgf_net":
        raise ValueError("train_sgf.py requires model.type=sgf_net")

    if args.validate_only:
        print("Config validation OK")
        return

    set_seed(int(cfg.get("seed", 0)), deterministic=bool(cfg.get("deterministic", True)))

    output_cfg = cfg.get("output", {})
    run_dir = make_run_dir(output_cfg.get("root", "outputs/runs"), output_cfg.get("run_name"))
    cfg.setdefault("output", {})["run_dir"] = run_dir
    save_resolved_config(run_dir, cfg)
    write_meta(run_dir, cfg)

    dataloaders = build_train_dataloaders(cfg, run_dir=run_dir)
    model = build_model_from_config(cfg).to(args.device)

    # Print model summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SGF-Net] Total params: {total:,}")
    print(f"[SGF-Net] Trainable params: {trainable:,}")
    print(f"[SGF-Net] Frozen params: {total - trainable:,}")

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
    print(f"[SGF-Net] Training complete. Run dir: {run_dir}")
    print(f"[SGF-Net] Best {summary['best_metric_name']}: {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")


if __name__ == "__main__":
    main()
