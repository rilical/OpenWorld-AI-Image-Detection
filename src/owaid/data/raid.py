"""RAID dataset wrapper.

Prefer Hugging Face loading first. If unavailable or unreachable, fall back to local
files under ``RAID_ROOT``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset

from .transforms import _to_tensor
from ..utils.paths import require_env


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
        root = Path(data_root or require_env("RAID_ROOT"))
        if root.exists():
            local_records = self._load_local_root(root)
            if local_records:
                return local_records

        try:
            from datasets import load_dataset

            ds = load_dataset("OwensLab/RAID", split=self.split)
            return [dict(r) for r in ds]
        except Exception as exc:
            raise RuntimeError(
                "RAID data is not available locally and Hugging Face fetch failed. "
                "Set RAID_ROOT or ensure OwensLab/RAID is accessible."
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
                        records.append({"path": str(path), "label": int(label), "id": str(path)})

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
                    records.append({"path": str(path), "label": int(label), "id": str(path)})

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
                "id": row.get("id", row.get("image_id", idx)),
                "source_dataset": "RAID",
                "split": self.split,
                "path": row.get("path"),
            },
        }
