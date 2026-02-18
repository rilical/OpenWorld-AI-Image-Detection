"""Data loading entry points and dataloader builders."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .commfor_small import CommunityForensicsSmallDataset
from .vct2 import VCT2Dataset
from .raid import RAIDDataset
from .aria import ARIADataset
from .transforms import build_clip_transform


def _extract_run_dir(cfg_dict: Dict[str, Any], data_cfg: Dict[str, Any]) -> str | None:
    if isinstance(data_cfg, dict) and data_cfg.get("run_dir"):
        return data_cfg.get("run_dir")

    output_cfg = cfg_dict.get("output", {})
    if isinstance(output_cfg, dict) and output_cfg.get("run_dir"):
        return output_cfg.get("run_dir")
    return None


def _save_split_indices(
    run_dir: str,
    train_indices: list[int],
    calib_indices: list[int],
    source: str = "commfor",
) -> None:
    if not run_dir:
        return
    path = Path(run_dir) / "calibration" / f"{source}_split_indices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_indices": train_indices,
                "calibration_indices": calib_indices,
            },
            f,
            indent=2,
        )


def _extract_cfg(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    return {}


def _build_hf_dataset(*, split: str, streaming: bool):
    from datasets import load_dataset

    return load_dataset("OwensLab/CommunityForensics-Small", split=split, streaming=streaming)


def build_commfor_dataloaders(cfg: Any) -> Dict[str, DataLoader]:
    """Load CommunityForensics-Small and return train/calibration/val dataloaders."""
    cfg_dict = _extract_cfg(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    seed = int(cfg_dict.get("seed", 42))
    calibration_fraction = float(data_cfg.get("calibration_fraction", 0.1))

    streaming = bool(data_cfg.get("streaming", False))
    if streaming:
        warnings.warn(
            "CommunityForensics-Small streaming mode cannot build deterministic train/calibration splits. "
            "Falling back to non-streaming load for reproducible split extraction."
        )
        streaming = False

    train_set = _build_hf_dataset(split="train", streaming=streaming)
    train_transform = build_clip_transform(cfg_dict, train=True)
    train_dataset = CommunityForensicsSmallDataset(train_set, transform=train_transform, split="train")

    n = len(train_dataset)
    if not (0.0 <= calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in [0.0, 1.0)")

    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_at = int((1.0 - calibration_fraction) * n)
    split_at = min(max(0, split_at), n)
    train_idx = indices[:split_at].astype(int).tolist()
    cal_idx = indices[split_at:].astype(int).tolist()

    run_dir = _extract_run_dir(cfg_dict, data_cfg)
    _save_split_indices(str(run_dir), train_idx, cal_idx)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    cal_dataset = CommunityForensicsSmallDataset(train_set, transform=train_transform, split="train")
    cal_loader = DataLoader(
        Subset(cal_dataset, cal_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    loaders: Dict[str, DataLoader] = {
        "train": train_loader,
        "cal": cal_loader,
        "calibration": cal_loader,
    }

    # Optional official validation split.
    try:
        val_set = _build_hf_dataset(split="validation", streaming=streaming)
        val_dataset = CommunityForensicsSmallDataset(
            val_set,
            transform=build_clip_transform(cfg_dict, train=False),
            split="validation",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        loaders["val"] = val_loader
    except Exception:
        # Keep robust if dataset has no validation split or offline mode is used.
        pass

    return loaders


def build_eval_dataloader(cfg: Any, dataset_name: str) -> DataLoader:
    """Build an evaluation dataloader for the requested dataset."""
    cfg_dict = _extract_cfg(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    transform = build_clip_transform(cfg_dict, train=False)

    name = (dataset_name or "").lower()
    if name in {"commfor", "commfor_small", "communityforensics-small", "communityforensics_small"}:
        split = data_cfg.get("split", "validation")
        ds = _build_hf_dataset(split=split, streaming=False)
        dataset = CommunityForensicsSmallDataset(ds, transform=transform, split=split)
    elif name == "vct2":
        dataset = VCT2Dataset(split=data_cfg.get("split", "test"), transform=transform, data_root=data_cfg.get("vct2_root"))
    elif name == "raid":
        dataset = RAIDDataset(split=data_cfg.get("split", "test"), transform=transform, data_root=data_cfg.get("raid_root"))
    elif name == "aria":
        dataset = ARIADataset(split=data_cfg.get("split", "test"), transform=transform, data_root=data_cfg.get("aria_root"))
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# Compatibility aliases expected by other modules in the scaffold.
def build_vct2_dataloader(cfg: Any) -> DataLoader:
    return build_eval_dataloader(cfg, "vct2")


def build_raid_dataloader(cfg: Any) -> DataLoader:
    return build_eval_dataloader(cfg, "raid")


def build_aria_dataloader(cfg: Any) -> DataLoader:
    return build_eval_dataloader(cfg, "aria")


__all__ = [
    "build_commfor_dataloaders",
    "build_eval_dataloader",
    "build_vct2_dataloader",
    "build_raid_dataloader",
    "build_aria_dataloader",
    "CommunityForensicsSmallDataset",
    "VCT2Dataset",
    "RAIDDataset",
    "ARIADataset",
]
