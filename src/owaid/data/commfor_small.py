"""CommunityForensics-Small dataset wrapper and split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset


class CommunityForensicsSmallDataset(Dataset):
    """Wrapper over a Hugging Face CommunityForensics split."""

    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        label_map: Optional[Dict[str, int]] = None,
        split: str | None = None,
        real_only: bool = False,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_map = label_map or {"real": 0, "human": 0, "ai": 1, "synth": 1, "synthetic": 1}
        self.split = split
        self.real_only = real_only

    def __len__(self) -> int:
        return len(self.dataset)

    def _extract_label(self, row: Dict[str, Any]) -> int:
        if self.real_only:
            return 0

        raw = row.get("label", row.get("target", row.get("y", 0)))
        if isinstance(raw, (bool, int)):
            return int(raw)
        if isinstance(raw, str):
            raw_key = raw.lower()
            if raw_key in self.label_map:
                return int(self.label_map[raw_key])
            if raw_key in {"ai", "synthetic", "fake", "1"}:
                return 1
            if raw_key in {"real", "human", "real_photo", "0"}:
                return 0
        raise ValueError(f"Could not parse label from sample: {row}")

    def _extract_meta(self, row: Dict[str, Any], index: int) -> Dict[str, Any]:
        return {
            "id": row.get("id", row.get("image_id", index)),
            "source_dataset": "CommunityForensics-Small",
            "split": self.split,
            "generator": row.get("generator"),
            "path": row.get("path"),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataset[index]
        if not isinstance(row, dict):
            # dataset row objects can be custom row wrappers in older datasets versions
            row = dict(row)

        image = row.get("image")
        if image is None and "img" in row:
            image = row["img"]
        if image is None and "jpeg" in row:
            image = row["jpeg"]

        if image is None:
            raise ValueError(f"Missing image in dataset row at index {index}")

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(image)
        elif torch.is_tensor(image):
            tensor = image.float()
        elif hasattr(image, "numpy"):
            tensor = torch.as_tensor(image.numpy())
            if tensor.ndim == 3 and tensor.shape[0] != 3:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.float()
        else:
            from .transforms import _to_tensor

            tensor = _to_tensor(image)

        return {
            "image": tensor,
            "label": self._extract_label(row),
            "meta": self._extract_meta(row, index),
        }
