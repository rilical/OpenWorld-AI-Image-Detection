"""ARIA real-only dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset

from .transforms import _to_tensor
from ..utils.paths import require_env


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
                "id": str(path),
                "source_dataset": "ARIA",
                "split": self.split,
                "path": str(path),
                "real_only": True,
            },
        }


__all__ = ["ARIADataset"]
