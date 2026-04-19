"""AI-Art-vs-Real dataset wrapper (Hemg/AI-Generated-vs-Real-Images-Datasets).

152k HF parquet dataset, labels ``AiArtData=0`` and ``RealArt=1`` at source.
We invert to the project convention ``real=0, fake=1`` so TPR/FPR stay aligned
with CommFor/RAID.

This is an out-of-distribution benchmark for CommFor-trained detectors: most
samples are stylized AI art (Midjourney, SD-art-style prompts) rather than the
photorealistic fakes CommFor-Small emphasises, so baselines that overfit to
photo artifacts should degrade.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from ..utils.paths import stable_sample_id


_SPLIT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "outputs" / "splits" / "aiart" / "split.json"
)


class AIArtDataset(Dataset):
    """Wrap Hemg/AI-Generated-vs-Real-Images-Datasets with project label scheme."""

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

        # Source: 0=AiArtData, 1=RealArt. Flip to project scheme (real=0, fake=1).
        src_label = int(row.get("label", 0))
        label = 1 - src_label

        sample_id = stable_sample_id("aiart", provided_id=None, split=self.split, index=index)
        return {
            "image": tensor,
            "label": label,
            "arch_label": -1,
            "domain_label": 1,  # 0=CommFor, 1=aiart
            "meta": {
                "id": str(sample_id),
                "source_dataset": "Hemg/AI-Generated-vs-Real-Images-Datasets",
                "split": str(self.split or ""),
                "generator": "MixedAIArt" if label == 1 else "Real",
                "path": "",
            },
        }


def load_aiart_with_split(
    split: str,
    transform: Optional[Callable[[Any], torch.Tensor]] = None,
    seed: int = 123,
) -> "AIArtDataset":
    """Load the 5000-sample aiart subset and restrict to one of the 80/20 shards.

    Parameters
    ----------
    split : {"train", "test"}
        Which shard of the stratified 80/20 split to return. "train" yields
        4000 samples (80%), "test" yields 1000 samples (20%).
    transform : callable, optional
        Image transform applied inside :class:`AIArtDataset`.
    seed : int, default=123
        Kept for signature compatibility — the split manifest is produced by
        ``scripts/create_aiart_split.py`` at seed=123 and is assumed to already
        reflect that seed. A mismatch against ``manifest["seed"]`` raises.

    Returns
    -------
    AIArtDataset
        Wrapped Hugging Face dataset restricted to the requested shard.

    Raises
    ------
    FileNotFoundError
        If ``outputs/splits/aiart/split.json`` is missing. Run
        ``scripts/create_aiart_split.py`` first.
    ValueError
        If ``split`` is not one of ``{"train", "test"}`` or the manifest seed
        disagrees with the requested seed.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    manifest_path = _SPLIT_MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"aiart split manifest not found at {manifest_path}. "
            "Run scripts/create_aiart_split.py first."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(
            f"aiart split manifest seed ({manifest.get('seed')}) does not match "
            f"requested seed ({seed}). Rebuild the split or use seed=123."
        )

    subset_indices = [int(i) for i in manifest["subset_indices"]]
    positions_key = "train_indices" if split == "train" else "test_indices"
    positions = [int(i) for i in manifest[positions_key]]

    from datasets import load_dataset

    source = manifest.get("source", "Hemg/AI-Generated-vs-Real-Images-Datasets")
    ds = load_dataset(source, split="train")
    ds = ds.select(subset_indices)
    ds = ds.select(positions)

    return AIArtDataset(ds, transform=transform, split=split)
