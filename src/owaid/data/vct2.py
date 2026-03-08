"""VCT2 benchmark wrapper.

Expected local structure:

1) Manifest mode (preferred)
   - Root directory contains one of:
     - ``{split}.csv`` with columns: ``path``, ``label``
     - ``{split}.json`` as JSON list/dicts containing ``path`` and ``label``

   Each row should also contain optional ``split``; the loader filters by the requested split.

2) Folder mode
   - ``root/{split}/real/*``
   - ``root/{split}/ai/*``

Environment variable ``VCT2_ROOT`` may be used when ``data_root`` is not provided.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import _to_tensor
from .transforms import build_clip_transform
from ..utils.paths import require_env, stable_sample_id


class VCT2Dataset(Dataset):
    """VCT2 image dataset using manifest files or split/class directories."""

    def __init__(
        self,
        split: str = "test",
        transform: Optional[Callable] = None,
        data_root: str | None = None,
    ):
        self.split = split
        self.transform = transform
        self.root = Path(data_root or require_env("VCT2_ROOT"))
        self.samples = self._index_samples()

        if not self.samples:
            raise RuntimeError(
                "VCT2 data not found. Set VCT2_ROOT to a populated directory with either "
                "`{split}.csv` / `{split}.json` manifest files or `root/{split}/{real,ai}` folders."
            )

    def _index_samples(self) -> list[dict[str, Any]]:
        # 1) Manifest mode
        for manifest in [f"{self.split}.csv", f"{self.split}.json"]:
            path = self.root / manifest
            if path.exists():
                samples = self._read_manifest(path)
                if samples:
                    return samples

        for manifest in ["manifest.csv", "manifest.json"]:
            path = self.root / manifest
            if path.exists():
                samples = self._read_manifest(path)
                if samples:
                    return samples

        # 2) Folder mode
        candidates: list[dict[str, Any]] = []
        split_dir = self.root / self.split
        for label_name, label in [("real", 0), ("ai", 1)]:
            class_dir = split_dir / label_name
            if not class_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
                for path in class_dir.glob(ext):
                    candidates.append({"path": path, "label": label})

        if not candidates:
            available = sorted(p.name for p in self.root.rglob("*") if p.is_file())[:20]
            available_text = ", ".join(str(x) for x in available)
            raise RuntimeError(
                "No usable VCT2 manifest or split directory structure found. "
                "Expected VCT2_ROOT/<split>/{real,ai}/* or a JSON/CSV manifest. "
                f"Configured VCT2_ROOT={self.root}. Example files: {available_text}"
            )

        return candidates

    def _read_manifest(self, manifest_path: Path):
        if manifest_path.suffix.lower() == ".csv":
            with manifest_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif manifest_path.suffix.lower() == ".json":
            with manifest_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                rows = payload.get("samples", payload.get(self.split, payload.get("data", [])))
            elif isinstance(payload, list):
                rows = payload
            else:
                raise RuntimeError(f"Unsupported manifest structure in {manifest_path}")
        else:
            raise RuntimeError(f"Unsupported manifest extension: {manifest_path.suffix}")

        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            split_name = str(row.get("split", self.split))
            if split_name != self.split:
                continue
            rel = str(row.get("path", "")).strip()
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = self.root / path
            label = row.get("label")
            if label is None:
                continue
            out.append({
                "path": path,
                "label": int(label),
                "generator": row.get("generator"),
                "group": row.get("group"),
            })
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        path = sample["path"]
        label = int(sample["label"])
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image) if self.transform else _to_tensor(image)
        return {
            "image": tensor,
            "label": label,
            "meta": {
                "id": stable_sample_id("vct2", path=path, root=self.root),
                "source_dataset": "VCT2",
                "split": self.split,
                "path": str(path),
                "generator": sample.get("generator"),
                "group": sample.get("group"),
            },
        }


@dataclass
class VCT2Config:
    root: str = ""


def build_vct2_dataloader(cfg: Dict[str, Any] | Any) -> DataLoader:
    """Build a VCT2 evaluation dataloader from config."""
    cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    transform = build_clip_transform(cfg_dict, train=False)
    dataset = VCT2Dataset(
        split=data_cfg.get("split", "test"),
        transform=transform,
        data_root=data_cfg.get("vct2_root"),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=max(0, int(data_cfg.get("num_workers", 2))),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


__all__ = ["VCT2Dataset", "VCT2Config", "build_vct2_dataloader"]
