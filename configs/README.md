# Configs

This directory contains the stable YAML entrypoints used by the repo scripts.

## Common structure

- `seed`, `deterministic`, `device`
- `output.root`, `output.run_name`
- `data.dataset`, `data.batch_size`, `data.num_workers`, `data.img_size`
- `data.calibration_fraction` for CommunityForensics-Small
- `data.transforms.*` for CLIP preprocessing and optional corruptions
- `model.type`, `model.clip.*`, `model.head.*`, `model.dire.*`
- `train.*` for optimization settings
- `calibration.temperature`, `calibration.conformal_alpha`, `calibration.conformal_method`
- `eval.save_predictions`

## Dataset path conventions

- CommunityForensics-Small uses Hugging Face `datasets` and respects `HF_HOME` / `HF_DATASETS_CACHE`.
- VCT2 uses `VCT2_ROOT` unless `data.vct2_root` is set.
- RAID uses Hugging Face first and falls back to `RAID_ROOT` or `data.raid_root`.
- ARIA uses `ARIA_ROOT` unless `data.aria_root` is set.

## Stable sample IDs

All dataset loaders should emit a stable `meta.id` that survives caching and prediction export:

- CommunityForensics-Small: dataset-provided `id` or `image_id`, otherwise a deterministic split/index fallback.
- Manifest datasets: `dataset_name:<relative_or_manifest_path>`.
- Folder datasets: `dataset_name:<relative_path_from_root>`.

These IDs are reused by prediction exports, split manifests, and residual caches.
