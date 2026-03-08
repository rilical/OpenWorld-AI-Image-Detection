"""RAID dataset wrapper.

Prefer Hugging Face loading first. If unavailable or unreachable, fall back to local
files under ``RAID_ROOT``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import _to_tensor
from .transforms import build_clip_transform
from ..utils.paths import stable_sample_id


class RAIDDataset(Dataset):
    """RAID benchmark dataset (HF or local filesystem fallback)."""

    def __init__(
        self,
        split: str = "test",
        transform: Optional[Callable] = None,
        data_root: str | None = None,
    ):
        self.transform = transform
        self.split = split
        self.samples = self._load_records(data_root)

        if not self.samples:
            raise RuntimeError(
                "RAID data not found. Set RAID_ROOT to a valid local dataset path "
                "or ensure OwensLab/RAID is accessible in Hugging Face datasets."
            )

    def _load_records(self, data_root: str | None):
        root = Path(data_root).expanduser().resolve() if data_root else None
        if root is not None and root.exists():
            local_records = self._load_local_root(root)
            if local_records:
                return local_records

        try:
            from datasets import load_dataset

            ds = load_dataset("OwensLab/RAID", split=self.split)
            return [dict(r) for r in ds]
        except Exception as exc:
            env_root = Path(Path.home())  # sentinel, replaced below if env exists
            env_root_value = None
            import os

            env_root_value = os.environ.get("RAID_ROOT")
            if env_root_value:
                env_root = Path(env_root_value).expanduser().resolve()
                if env_root.exists():
                    local_records = self._load_local_root(env_root)
                    if local_records:
                        return local_records
            raise RuntimeError(
                "RAID data is not available from Hugging Face and no usable local fallback was found. "
                "Set RAID_ROOT or pass data.raid_root to a populated local dataset path, "
                "or ensure OwensLab/RAID is accessible."
            ) from exc

    def _load_local_root(self, root: Path):
        records: list[Dict[str, Any]] = []

        # Preferred split + class directory layout.
        split_root = root / self.split
        for class_name, label in [("real", 0), ("ai", 1), ("fake", 1)]:
            class_dir = split_root / class_name
            if class_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
                    for path in class_dir.rglob(ext):
                        records.append({"path": str(path), "label": int(label), "root": str(root)})

        if records:
            return records

        # Manifest fallback.
        for manifest in [root / "data.json", root / "manifest.json", root / "raid.json"]:
            if manifest.exists():
                with manifest.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload = payload.get("samples", payload.get(self.split, []))
                if isinstance(payload, list):
                    return [dict(item) for item in payload]
                break

        # Class directory fallback (no split nesting).
        for class_name, label in [("real", 0), ("ai", 1), ("fake", 1)]:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
                for path in class_dir.rglob(ext):
                    records.append({"path": str(path), "label": int(label), "root": str(root)})

        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        if "image" in row and row["image"] is not None:
            image = row["image"]
            if hasattr(image, "convert"):
                image = image.convert("RGB")
        else:
            image = Image.open(row["path"]).convert("RGB")

        tensor = self.transform(image) if self.transform else _to_tensor(image)
        raw_label = row.get("label", row.get("target", 0))
        label = int(raw_label) if isinstance(raw_label, (int, bool)) else 1 if raw_label else 0

        return {
            "image": tensor,
            "label": label,
            "meta": {
                "id": stable_sample_id(
                    "raid",
                    provided_id=row.get("id", row.get("image_id")),
                    path=row.get("path"),
                    split=self.split,
                    index=idx,
                    root=row.get("root"),
                ),
                "source_dataset": "RAID",
                "split": self.split,
                "path": row.get("path"),
                "generator": row.get("generator"),
                "pair_id": row.get("pair_id", row.get("source_id")),
                "variant": row.get("variant"),
                "is_adversarial": row.get("is_adversarial"),
            },
        }


def build_raid_dataloader(cfg: Dict[str, Any] | Any) -> DataLoader:
    """Build a RAID evaluation dataloader from config."""
    cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    transform = build_clip_transform(cfg_dict, train=False)
    dataset = RAIDDataset(
        split=data_cfg.get("split", "test"),
        transform=transform,
        data_root=data_cfg.get("raid_root"),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=max(0, int(data_cfg.get("num_workers", 2))),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


__all__ = ["RAIDDataset", "build_raid_dataloader"]
