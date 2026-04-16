# Datasets

## CommunityForensics-Small (Training & Validation)

- **Source**: Hugging Face `OwensLab/CommunityForensics-Small`
- **Download**: Automatic via `datasets` library (no manual steps)
- **Size**: ~278K real + ~278K AI images
- **License**: CC BY-NC-SA (non-commercial)
- **Env vars**: `HF_HOME` / `HF_DATASETS_CACHE` (optional, controls cache location)
- **Splits used**: `train` (split into train_fit + calibration), `validation`
- **Loader**: `src/owaid/data/commfor_small.py`
- **Stable sample id**: use dataset `id` or `image_id` when present, otherwise `commfor:<split>:<index>`
- **Streaming caveat**: streaming mode is not used for reproducible train/calibration splitting; loaders fall back to non-streaming mode when a deterministic split manifest is required.

## VCT2 (Open-World Evaluation)

- **Source**: VCT2 benchmark (~130K images from unseen generators)
- **Download**: Manual acquisition required
- **Env var**: `VCT2_ROOT` (required)
- **Config override**: `data.vct2_root`
- **Expected structure** (either mode):
  - Manifest mode: `$VCT2_ROOT/{split}.csv` or `$VCT2_ROOT/{split}.json` with columns `path`, `label`
  - Folder mode: `$VCT2_ROOT/{split}/real/*`, `$VCT2_ROOT/{split}/ai/*`
- **Loader**: `src/owaid/data/vct2.py`
- **Stable sample id**: `vct2:<relative-path-or-manifest-path>`
- **Note**: manual download is required.

## RAID (Adversarial Evaluation)

- **Source**: RAID adversarial benchmark (~72K examples)
- **Download**: Hugging Face `aimagelab/RAID` (preferred) or local files
- **Env var**: `RAID_ROOT` (optional for local mode)
- **Config override**: `data.raid_root`
- **Expected structure** (local):
  - Split/class dirs: `$RAID_ROOT/{split}/{real,ai,fake}/*`
  - Or manifest: `$RAID_ROOT/data.json` / `manifest.json` / `raid.json`
- **Loader**: `src/owaid/data/raid.py`
- **Stable sample id**: `raid:<relative-path-or-manifest-path>`
- **Note**: if Hugging Face access fails, configure `RAID_ROOT` explicitly.

## ARIA (Real-Only Evaluation)

- **Source**: AdvAIArt project (real-only images for false positive testing)
- **Download**: https://github.com/AdvAIArtProject/AdvAIArt
- **Env var**: `ARIA_ROOT` (required)
- **Config override**: `data.aria_root`
- **Expected structure**: `$ARIA_ROOT/{split}/*.{jpg,png,...}` or `$ARIA_ROOT/*.{jpg,png,...}`
- **Note**: All images are real (label=0). Used to measure false positive rate.
- **Loader**: `src/owaid/data/aria.py`
- **Stable sample id**: `aria:<relative-path-from-root>`

## Stable sample-id policy

Every dataset loader should emit `meta.id` using a deterministic, cache-safe rule:

- Prefer a dataset-provided immutable sample identifier.
- Otherwise derive `dataset_name:<relative-path-from-dataset-root>`.
- If neither exists, fall back to `dataset_name:<split>:<index>`.

This ID is the primary key for prediction exports, split manifests, and residual-cache files.
