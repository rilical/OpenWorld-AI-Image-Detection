"""One-shot script to build a reproducible 80/20 stratified split of the
5000-sample aiart subset used across the project.

Writes ``outputs/splits/aiart/split.json`` with:
  - subset_indices : 5000 indices into the full 152k Hemg dataset (seed=123)
  - train_indices  : 4000 positions within [0, 5000) — the 80% train shard
  - test_indices   : 1000 positions within [0, 5000) — the 20% holdout
  - sha256         : integrity hash over sorted(train) + sorted(test)

Reproduces the seeding used in ``src/owaid/data/__init__.py``
(``build_eval_dataloader`` aiart branch) so every downstream component
operates on the same 5000 rows.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "HF_HOME",
    "/ocean/projects/cis250202p/gyar/personal/dl/OpenWorld-AI-Image-Detection/.cache/huggingface",
)

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()


SUBSET_SIZE = 5000
SEED = 123
SOURCE = "Hemg/AI-Generated-vs-Real-Images-Datasets"
OUT_PATH = ROOT_DIR / "outputs" / "splits" / "aiart" / "split.json"


def _stratified_split(labels: np.ndarray, test_size: float, seed: int) -> tuple[list[int], list[int]]:
    """80/20 stratified split of positions in [0, len(labels)).

    Uses ``sklearn.model_selection.StratifiedShuffleSplit`` when available
    and falls back to a manual per-class split otherwise (matching the
    pattern already used in ``build_commfor_dataloaders``).
    """
    indices = np.arange(len(labels))
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx_arr, test_idx_arr = next(splitter.split(indices, labels))
        return train_idx_arr.tolist(), test_idx_arr.tolist()
    except ImportError:
        rng = np.random.default_rng(seed)
        train_idx: list[int] = []
        test_idx: list[int] = []
        for c in np.unique(labels):
            c_indices = indices[labels == c].copy()
            rng.shuffle(c_indices)
            n_test = int(round(test_size * len(c_indices)))
            test_idx.extend(c_indices[:n_test].tolist())
            train_idx.extend(c_indices[n_test:].tolist())
        return train_idx, test_idx


def _sha256_of_splits(train_indices: list[int], test_indices: list[int]) -> str:
    combined = sorted(train_indices) + sorted(test_indices)
    data = json.dumps(combined).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    from datasets import load_dataset

    print(f"Loading {SOURCE} ...")
    ds = load_dataset(SOURCE, split="train")
    print(f"Full dataset size: {len(ds)}")

    # Reproduce the aiart subset selection used in build_eval_dataloader.
    rng = np.random.default_rng(SEED)
    subset_idx = rng.choice(len(ds), size=SUBSET_SIZE, replace=False)
    subset_indices: list[int] = [int(i) for i in subset_idx.tolist()]
    print(f"Sampled {len(subset_indices)} indices with seed={SEED}")

    # Read labels for only the selected subset. The source convention is
    # AiArtData=0 / RealArt=1; AIArtDataset flips to real=0, fake=1 at load
    # time, but for stratification purposes the source labels are fine —
    # we only need class membership to produce a balanced split.
    subset_ds = ds.select(subset_indices)
    if "label" in subset_ds.column_names:
        raw_labels = subset_ds["label"]
    else:
        raise RuntimeError(
            f"Expected 'label' column in {SOURCE}; columns={subset_ds.column_names}"
        )
    labels = np.asarray([int(x) for x in raw_labels], dtype=np.int64)

    train_positions, test_positions = _stratified_split(labels, test_size=0.2, seed=SEED)
    train_positions = [int(i) for i in train_positions]
    test_positions = [int(i) for i in test_positions]

    assert len(train_positions) + len(test_positions) == SUBSET_SIZE
    assert set(train_positions).isdisjoint(test_positions)

    manifest = {
        "protocol_version": "1.0",
        "seed": SEED,
        "source": SOURCE,
        "subset_size": SUBSET_SIZE,
        "subset_indices": subset_indices,
        "train_indices": train_positions,
        "test_indices": test_positions,
        "sha256": _sha256_of_splits(train_positions, test_positions),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote split manifest to {OUT_PATH}")

    # Label counts per split (source-label space: 0=AiArtData, 1=RealArt).
    train_labels = labels[train_positions]
    test_labels = labels[test_positions]
    print("Label counts (source convention: 0=AiArtData, 1=RealArt):")
    print(
        f"  train: n={len(train_positions)}  "
        f"AiArtData={(train_labels == 0).sum()}  RealArt={(train_labels == 1).sum()}"
    )
    print(
        f"  test : n={len(test_positions)}  "
        f"AiArtData={(test_labels == 0).sum()}  RealArt={(test_labels == 1).sum()}"
    )
    print("(Project convention flips these: real=0, fake=1.)")


if __name__ == "__main__":
    main()
