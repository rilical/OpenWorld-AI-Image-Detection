"""Data loading entry points and dataloader builders."""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate


def _collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return default_collate(batch)

from .commfor_small import CommunityForensicsSmallDataset, CommunityForensicsSmallIterableDataset
from .vct2 import VCT2Dataset
from .raid import RAIDDataset, RAIDStreamingDataset
from .aria import ARIADataset
from .aiart import AIArtDataset, load_aiart_with_split
from .cifake import CIFAKEDataset
from .transforms import build_clip_transform


def _islice_dataset(iterable, n: int):
    """Yield the first *n* items from an iterable (streaming HF dataset)."""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item


def _extract_run_dir(cfg_dict: Dict[str, Any], data_cfg: Dict[str, Any]) -> str | None:
    if isinstance(data_cfg, dict) and data_cfg.get("run_dir"):
        return data_cfg.get("run_dir")

    output_cfg = cfg_dict.get("output", {})
    if isinstance(output_cfg, dict) and output_cfg.get("run_dir"):
        return output_cfg.get("run_dir")
    return None


def _compute_indices_hash(train_indices: list[int], calib_indices: list[int]) -> str:
    """Compute SHA-256 hash over sorted indices for integrity verification."""
    combined = sorted(train_indices) + sorted(calib_indices)
    data = json.dumps(combined).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _save_split_manifest(
    run_dir: str,
    train_indices: list[int],
    calib_indices: list[int],
    seed: int,
    calibration_fraction: float,
    total_n: int,
) -> None:
    """Save split manifest with full provenance for reproducibility."""
    if not run_dir:
        return
    manifest = {
        "protocol_version": "1.0",
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "total_n": total_n,
        "train_indices": train_indices,
        "calibration_indices": calib_indices,
        "sha256": _compute_indices_hash(train_indices, calib_indices),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(run_dir) / "splits" / "split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also write legacy format for backward compatibility
    legacy_path = Path(run_dir) / "calibration" / "commfor_split_indices.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    with legacy_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"train_indices": train_indices, "calibration_indices": calib_indices},
            f, indent=2,
        )


def _load_split_manifest(run_dir: str) -> Dict[str, Any] | None:
    """Load existing split manifest if available."""
    path = Path(run_dir) / "splits" / "split_manifest.json"
    if not path.exists():
        path = Path(run_dir) / "calibration" / "split_manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    # Verify integrity
    expected_hash = _compute_indices_hash(
        manifest["train_indices"], manifest["calibration_indices"]
    )
    if manifest.get("sha256") != expected_hash:
        warnings.warn("Split manifest hash mismatch — recomputing split.")
        return None
    return manifest


def _extract_cfg(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    return {}


def _build_hf_dataset(*, split: str, streaming: bool, training=True):
    from datasets import load_dataset
    
    # training
    if training:
        return load_dataset("OwensLab/CommunityForensics-Small", split=split, streaming=streaming)
    # evaluation
    else: 
        return load_dataset("OwensLab/CommunityForensics-Eval", split=split, streaming=streaming)


def _extract_labels_fast(hf_dataset, label_map: Dict[str, int] | None = None) -> np.ndarray:
    """Extract labels from raw HF dataset without decoding images.

    Much faster and less RAM-intensive than iterating through the wrapped
    Dataset (which loads and transforms every image).
    """
    label_map = label_map or {"real": 0, "human": 0, "ai": 1, "synth": 1, "synthetic": 1}

    # HF datasets expose column data directly — no image decode needed.
    if hasattr(hf_dataset, "column_names") and "label" in hf_dataset.column_names:
        raw_labels = hf_dataset["label"]
    elif hasattr(hf_dataset, "column_names") and "target" in hf_dataset.column_names:
        raw_labels = hf_dataset["target"]
    else:
        # Fallback: iterate rows but only read label fields
        raw_labels = []
        for row in hf_dataset:
            raw_labels.append(row.get("label", row.get("target", row.get("y", 0))))

    out = []
    for raw in raw_labels:
        if isinstance(raw, (bool, int)):
            out.append(int(raw))
        elif isinstance(raw, str):
            raw_key = raw.lower()
            if raw_key in label_map:
                out.append(int(label_map[raw_key]))
            elif raw_key in {"ai", "synthetic", "fake", "1"}:
                out.append(1)
            elif raw_key in {"real", "human", "real_photo", "0"}:
                out.append(0)
            else:
                raise ValueError(f"Could not parse label: {raw!r}")
        else:
            out.append(int(raw))
    return np.array(out, dtype=np.int64)


def build_commfor_dataloaders(cfg: Any, run_dir: str | None = None) -> Dict[str, DataLoader]:
    """Load CommunityForensics-Small and return train/calibration/val dataloaders."""
    cfg_dict = _extract_cfg(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    seed = int(cfg_dict.get("seed", 42))
    calibration_fraction = float(data_cfg.get("calibration_fraction", 0.1))
    max_train_samples = data_cfg.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    max_eval_samples = data_cfg.get("max_eval_samples")
    if max_eval_samples is not None:
        max_eval_samples = int(max_eval_samples)
    shuffle_buffer = int(data_cfg.get("shuffle_buffer", 100))

    streaming = bool(data_cfg.get("streaming", False))
    if streaming and max_train_samples is None:
        warnings.warn(
            "CommunityForensics-Small streaming mode cannot build deterministic "
            "train/calibration splits without max_train_samples. "
            "Falling back to non-streaming load."
        )
        streaming = False

    train_set = _build_hf_dataset(split="train", streaming=streaming, training=True)

    if streaming:
        # Streaming path: shuffle then take a fixed number of samples,
        # materialise into a regular HF dataset for indexing.
        train_set = train_set.shuffle(seed=seed, buffer_size=shuffle_buffer)
        rows = list(_islice_dataset(train_set, max_train_samples))
        from datasets import Dataset as HFDataset
        train_set = HFDataset.from_list(rows)
    elif max_train_samples is not None and max_train_samples < len(train_set):
        # Non-streaming but limited: use stratified sampling to preserve class balance.
        all_labels = _extract_labels_fast(train_set)
        unique_classes = np.unique(all_labels)
        rng = np.random.default_rng(seed)
        if len(unique_classes) >= 2:
            # Stratified: sample proportionally from each class
            subset_idx = []
            for c in unique_classes:
                c_indices = np.where(all_labels == c)[0]
                n_c = max(1, int(round(len(c_indices) / len(all_labels) * max_train_samples)))
                n_c = min(n_c, len(c_indices))
                subset_idx.extend(rng.choice(c_indices, size=n_c, replace=False).tolist())
            # Trim or pad to exactly max_train_samples
            rng.shuffle(subset_idx)
            subset_idx = subset_idx[:max_train_samples]
        else:
            subset_idx = rng.choice(len(train_set), size=max_train_samples, replace=False).tolist()
        train_set = train_set.select(subset_idx)

    train_transform = build_clip_transform(cfg_dict, train=True)
    train_dataset = CommunityForensicsSmallDataset(train_set, transform=train_transform, split="train")

    n = len(train_dataset)
    if not (0.0 <= calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in [0.0, 1.0)")

    run_dir = run_dir or _extract_run_dir(cfg_dict, data_cfg)

    # Try loading existing manifest for reproducibility
    manifest = _load_split_manifest(str(run_dir)) if run_dir else None
    if manifest and manifest.get("total_n") == n and manifest.get("seed") == seed:
        train_idx = manifest["train_indices"]
        cal_idx = manifest["calibration_indices"]
    else:
        # Extract labels efficiently (no image decode)
        labels = _extract_labels_fast(train_set)

        # Sanity check: both classes must be present
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Training data contains only labels {unique_labels.tolist()}. "
                f"Both classes (0=real, 1=AI) are required. "
                f"Try increasing max_train_samples or check dataset integrity."
            )

        indices = np.arange(n)

        if calibration_fraction == 0.0:
            train_idx = indices.tolist()
            cal_idx = []
        else:
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=calibration_fraction, random_state=seed
                )
                train_idx_arr, cal_idx_arr = next(splitter.split(indices, labels))
                train_idx = train_idx_arr.tolist()
                cal_idx = cal_idx_arr.tolist()
            except ImportError:
                # Fallback: manual stratified split
                rng = np.random.default_rng(seed)
                train_idx, cal_idx = [], []
                for c in np.unique(labels):
                    c_indices = indices[labels == c]
                    rng.shuffle(c_indices)
                    split_at = int((1.0 - calibration_fraction) * len(c_indices))
                    train_idx.extend(c_indices[:split_at].tolist())
                    cal_idx.extend(c_indices[split_at:].tolist())

        _save_split_manifest(
            str(run_dir), train_idx, cal_idx,
            seed=seed, calibration_fraction=calibration_fraction, total_n=n,
        )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate_skip_none,
    )

    eval_transform = build_clip_transform(cfg_dict, train=False)
    cal_dataset = CommunityForensicsSmallDataset(train_set, transform=eval_transform, split="train")
    cal_loader = DataLoader(
        Subset(cal_dataset, cal_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate_skip_none,
    )

    loaders: Dict[str, DataLoader] = {
        "train_fit": train_loader,
        "train": train_loader,
        "cal": cal_loader,
        "calibration": cal_loader,
    }

    # Optional official validation split.
    try:
        val_set = _build_hf_dataset(split="validation", streaming=False)
        if max_eval_samples is not None and max_eval_samples < len(val_set):
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(len(val_set), size=max_eval_samples, replace=False)
            val_set = val_set.select(subset_idx.tolist())
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
            collate_fn=_collate_skip_none,
        )
        loaders["val"] = val_loader
    except Exception:
        # Keep robust if dataset has no validation split or offline mode is used.
        pass

    return loaders


def build_mixed_train_dataloader(cfg: Any, run_dir: str | None = None) -> Dict[str, DataLoader]:
    """Build a 50/50 CommFor + aiart-train mixed training dataloader.

    Returns keys ``"train_fit"``, ``"train"``, ``"cal"``, ``"calibration"`` and
    (when CommFor exposes a validation split) ``"val"``. The interface matches
    :func:`build_commfor_dataloaders` so training scripts can swap it in without
    further changes.

    Mix strategy
    ------------
    ``train_fit`` / ``train`` is a :class:`ConcatDataset` of the CommFor
    training Subset (as produced by :func:`build_commfor_dataloaders`) and the
    aiart 80%-train Subset produced by
    :func:`owaid.data.aiart.load_aiart_with_split`. A
    :class:`WeightedRandomSampler` assigns equal **total** probability mass to
    each domain, so a batch is ~50/50 CommFor/aiart in expectation. Each sample
    is tagged with a ``domain_label`` (0=CommFor, 1=aiart) in its dict.

    Calibration
    -----------
    ``cal`` / ``calibration`` is the CommFor calibration loader. aiart has no
    "correct" calibration distribution for CommFor-trained detectors, and we
    want conformal coverage guarantees on the in-distribution shard.

    Validation
    ----------
    ``val`` is the CommFor validation loader (ID val), inherited from
    :func:`build_commfor_dataloaders` when available.
    """
    cfg_dict = _extract_cfg(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    dataset_name = str(data_cfg.get("dataset", "")).lower()
    if dataset_name != "commfor_aiart_mix":
        raise ValueError(
            "build_mixed_train_dataloader requires data.dataset='commfor_aiart_mix', "
            f"got {dataset_name!r}"
        )

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))

    # Delegate to the CommFor builder for the ID pieces. It already handles
    # stratified splits, calibration, manifest writing, and optional val.
    commfor_loaders = build_commfor_dataloaders(cfg, run_dir=run_dir)

    # Extract the already-sliced CommFor training Subset.
    commfor_train_subset = commfor_loaders["train"].dataset
    if not isinstance(commfor_train_subset, Subset):
        # Safety net — current build_commfor_dataloaders always returns a Subset.
        raise RuntimeError(
            "Expected build_commfor_dataloaders to return a Subset for the train loader; "
            f"got {type(commfor_train_subset).__name__}"
        )

    # Build the aiart 80% train shard with the same CLIP training transforms.
    train_transform = build_clip_transform(cfg_dict, train=True)
    aiart_train_dataset = load_aiart_with_split("train", transform=train_transform)

    n_commfor = len(commfor_train_subset)
    n_aiart = len(aiart_train_dataset)
    if n_commfor == 0 or n_aiart == 0:
        raise ValueError(
            f"Empty shard in mixed dataloader (n_commfor={n_commfor}, n_aiart={n_aiart})."
        )

    # Equal total probability mass per domain: each CommFor sample gets 1/n_commfor
    # and each aiart sample gets 1/n_aiart (summing to 1.0 per domain).
    weights = torch.cat(
        [
            torch.full((n_commfor,), 1.0 / n_commfor, dtype=torch.double),
            torch.full((n_aiart,), 1.0 / n_aiart, dtype=torch.double),
        ]
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n_commfor + n_aiart,
        replacement=True,
    )

    mixed_train_dataset = ConcatDataset([commfor_train_subset, aiart_train_dataset])
    mixed_train_loader = DataLoader(
        mixed_train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate_skip_none,
    )

    loaders: Dict[str, DataLoader] = {
        "train_fit": mixed_train_loader,
        "train": mixed_train_loader,
        "cal": commfor_loaders["cal"],
        "calibration": commfor_loaders["calibration"],
    }
    if "val" in commfor_loaders:
        loaders["val"] = commfor_loaders["val"]

    return loaders


def build_train_dataloaders(cfg: Any, run_dir: str | None = None) -> Dict[str, DataLoader]:
    """Dispatch to the correct training dataloader builder based on ``data.dataset``.

    - ``commfor_small`` -> :func:`build_commfor_dataloaders`
    - ``commfor_aiart_mix`` -> :func:`build_mixed_train_dataloader`
    """
    cfg_dict = _extract_cfg(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict) if isinstance(cfg_dict, dict) else {}
    name = str(data_cfg.get("dataset", "commfor_small")).lower() if isinstance(data_cfg, dict) else "commfor_small"
    if name == "commfor_small":
        return build_commfor_dataloaders(cfg, run_dir=run_dir)
    if name == "commfor_aiart_mix":
        return build_mixed_train_dataloader(cfg, run_dir=run_dir)
    raise ValueError(f"Unknown training dataset: {name!r}")


def build_eval_dataloader(cfg: Any, dataset_name: str) -> DataLoader:
    """Build an evaluation dataloader for the requested dataset."""
    cfg_dict = _extract_cfg(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    seed = int(cfg_dict.get("seed", 42))
    max_eval_samples = data_cfg.get("max_eval_samples")
    if max_eval_samples is not None:
        max_eval_samples = int(max_eval_samples)
    transform = build_clip_transform(cfg_dict, train=False)

    name = (dataset_name or "").lower()
    if name in {"commfor", "commfor_small", "communityforensics-small", "communityforensics_small"}:
        split = data_cfg.get("split", "validation")
        eval_streaming = bool(data_cfg.get("streaming", False))
        eval_training = data_cfg.get("eval_dataset", "training") == "training"
        ds = _build_hf_dataset(split=split, streaming=eval_streaming, training=eval_training)
        if eval_streaming:
            rows = list(_islice_dataset(ds, max_eval_samples or 2000))
            from datasets import Dataset as HFDataset
            ds = HFDataset.from_list(rows)
        elif max_eval_samples is not None and max_eval_samples < len(ds):
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(len(ds), size=max_eval_samples, replace=False)
            ds = ds.select(subset_idx.tolist())
        dataset = CommunityForensicsSmallDataset(ds, transform=transform, split=split)
    elif name == "vct2":
        dataset = VCT2Dataset(split=data_cfg.get("split", "test"), transform=transform, data_root=data_cfg.get("vct2_root"))
    elif name == "raid":
        if bool(data_cfg.get("streaming", False)):
            dataset = RAIDStreamingDataset(
                split=data_cfg.get("split", "test"),
                transform=transform,
                max_samples=max_eval_samples,
            )
            num_workers = 0
        else:
            dataset = RAIDDataset(split=data_cfg.get("split", "test"), transform=transform, data_root=data_cfg.get("raid_root"))
    elif name == "aria":
        dataset = ARIADataset(split=data_cfg.get("split", "test"), transform=transform, data_root=data_cfg.get("aria_root"))
    elif name in {"aiart", "ai_art"}:
        aiart_split = data_cfg.get("split")
        if aiart_split in {"train", "test"}:
            dataset = load_aiart_with_split(aiart_split, transform=transform)
        else:
            from datasets import load_dataset
            ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
            if max_eval_samples is not None and max_eval_samples < len(ds):
                rng = np.random.default_rng(seed)
                subset_idx = rng.choice(len(ds), size=max_eval_samples, replace=False)
                ds = ds.select(subset_idx.tolist())
            dataset = AIArtDataset(ds, transform=transform, split="train")
    elif name in {"cifake"}:
        from datasets import load_dataset
        split = data_cfg.get("split", "test")
        ds = load_dataset("dragonintelligence/CIFAKE-image-dataset", split=split)
        if max_eval_samples is not None and max_eval_samples < len(ds):
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(len(ds), size=max_eval_samples, replace=False)
            ds = ds.select(subset_idx.tolist())
        dataset = CIFAKEDataset(ds, transform=transform, split=split)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate_skip_none,
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
    "build_mixed_train_dataloader",
    "build_train_dataloaders",
    "build_eval_dataloader",
    "build_vct2_dataloader",
    "build_raid_dataloader",
    "build_aria_dataloader",
    "CommunityForensicsSmallDataset",
    "VCT2Dataset",
    "RAIDDataset",
    "AIArtDataset",
    "ARIADataset",
    "CIFAKEDataset",
    "load_aiart_with_split",
]
