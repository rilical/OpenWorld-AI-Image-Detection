"""CIFAKE dataset wrapper (dragonintelligence/CIFAKE-image-dataset).

120k 32x32 RGB HF parquet dataset, labels ``FAKE=0`` and ``REAL=1`` at source.
We invert to the project convention ``real=0, fake=1`` so TPR/FPR stay aligned
with CommFor/RAID/AIArt.

This is an out-of-distribution benchmark for CommFor-trained detectors:
all images are tiny (32x32, upscaled to 224 by the CLIP preprocess) and the
"real" half comes from CIFAR-10, so baselines trained on larger photorealistic
content should degrade. Always treated as OOD in our pipeline
(``domain_label=1``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from ..utils.paths import stable_sample_id


class CIFAKEDataset(Dataset):
    """Wrap dragonintelligence/CIFAKE-image-dataset with project label scheme."""

    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        split: str | None = None,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataset[index]
        if not isinstance(row, dict):
            row = dict(row)

        image = row.get("image")
        if image is None:
            return None

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            from .transforms import _to_tensor
            tensor = _to_tensor(image)

        # Source: 0=FAKE, 1=REAL. Flip to project scheme (real=0, fake=1).
        src_label = int(row.get("label", 0))
        label = 1 - src_label

        sample_id = stable_sample_id("cifake", provided_id=None, split=self.split, index=index)
        return {
            "image": tensor,
            "label": label,
            "arch_label": -1,
            "domain_label": 1,
            "meta": {
                "id": str(sample_id),
                "source_dataset": "CIFAKE",
                "split": str(self.split or ""),
                "generator": "CIFAKE-fake" if label == 1 else "CIFAR-10-real",
                "path": "",
            },
        }
