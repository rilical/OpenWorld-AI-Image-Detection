"""Populate the residual-feature cache used by the optional DIRE path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from owaid.data import build_commfor_dataloaders, build_eval_dataloader
from owaid.models.dire_residual import ResidualEncoder
from owaid.utils import load_yaml, merge_cli_overrides, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache residual features for DIRE-style fusion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", nargs="*", default=None, help="Split names to cache")
    parser.add_argument("--opts", nargs="*")
    return parser.parse_args()


def _resolve_splits(cfg: dict, requested_splits):
    if requested_splits:
        return requested_splits
    dataset_name = cfg.get("data", {}).get("dataset", "commfor_small")
    if dataset_name == "commfor_small":
        return ["train_fit", "calibration", "val"]
    return [cfg.get("data", {}).get("split", "test")]


def main() -> None:
    args = parse_args()
    cfg = merge_cli_overrides(load_yaml(args.config), args.opts)
    set_seed(int(cfg.get("seed", 0)), deterministic=bool(cfg.get("deterministic", True)))

    cache_dir = args.cache_dir or cfg.get("model", {}).get("dire", {}).get("cache_dir")
    if not cache_dir:
        raise ValueError("cache_residuals.py requires --cache-dir or model.dire.cache_dir")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.get("data", {}).get("dataset", "commfor_small")
    splits = _resolve_splits(cfg, args.split)
    encoder = ResidualEncoder(cache_dir=str(cache_path)).to(args.device)

    cached = 0
    if dataset_name == "commfor_small":
        loaders = build_commfor_dataloaders(cfg, run_dir=args.run)
        selected = {split: loaders[split] for split in splits if split in loaders}
    else:
        selected = {}
        for split in splits:
            split_cfg = dict(cfg)
            split_cfg.setdefault("data", {}).update({"split": split})
            selected[split] = build_eval_dataloader(split_cfg, dataset_name)

    for split_name, loader in selected.items():
        for batch in loader:
            sample_ids = [meta["id"] for meta in batch["meta"]]
            images = batch["image"].to(args.device)
            encoder(images, sample_ids=sample_ids)
            cached += len(sample_ids)

    write_json(
        str(cache_path / "cache_meta.json"),
        {
            "cache_dir": str(cache_path),
            "dataset": dataset_name,
            "splits": list(selected.keys()),
            "cached_samples": cached,
            "placeholder_residual_encoder": True,
            "run_dir": args.run,
        },
    )
    print(f"Cached residual features for {cached} samples in {cache_path}")


if __name__ == "__main__":
    main()
