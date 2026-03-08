"""ARIA real-only dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import _to_tensor
from .transforms import build_clip_transform
from ..utils.paths import require_env, stable_sample_id


class ARIADataset(Dataset):
    """ARIADataset is real-only and always emits label ``0``."""

    def __init__(
        self,
        split: str = "test",
        transform: Optional[Callable] = None,
        data_root: str | None = None,
    ):
        self.split = split
        self.transform = transform
        self.data_root = Path(data_root or require_env("ARIA_ROOT"))
        self.samples = self._index_files()

        if not self.samples:
            raise RuntimeError(
                "ARIA data not found. Set ARIA_ROOT to the ARIA image directory and expect class-agnostic real-only images under split folders."
            )

    def _index_files(self):
        split_root = self.data_root / self.split
        search_roots = [split_root] if split_root.exists() else [self.data_root]
        samples = []
        for root in search_roots:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
                for path in root.rglob(ext):
                    samples.append(path)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img) if self.transform else _to_tensor(img)
        return {
            "image": tensor,
            "label": 0,
            "meta": {
                "id": stable_sample_id("aria", path=path, root=self.data_root),
                "source_dataset": "ARIA",
                "split": self.split,
                "path": str(path),
                "real_only": True,
            },
        }


def build_aria_dataloader(cfg: Dict[str, Any] | Any) -> DataLoader:
    """Build an ARIA evaluation dataloader from config."""
    cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    transform = build_clip_transform(cfg_dict, train=False)
    dataset = ARIADataset(
        split=data_cfg.get("split", "test"),
        transform=transform,
        data_root=data_cfg.get("aria_root"),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=max(0, int(data_cfg.get("num_workers", 2))),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


__all__ = ["ARIADataset", "build_aria_dataloader"]
