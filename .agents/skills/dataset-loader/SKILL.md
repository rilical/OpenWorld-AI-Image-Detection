---
name: dataset-loader
description: Use when adding or changing dataset loaders, transforms, split manifests, sample-id policy, or env-var based dataset access for CommunityForensics-Small, VCT2, RAID, ARIA, or similar image datasets. Do not use for model, calibration, or plotting work.
---

# Dataset Loader

## Use when

- Adding a new dataset under `src/owaid/data/`.
- Changing transform builders, manifest generation, or stable sample IDs.
- Tightening missing-data errors or env-var path handling.

## Do not use when

- The task is about training logic, model architecture, calibration math, or report generation.

## Workflow

1. Keep dataset parsing in `src/owaid/data/` and keep script code thin.
2. Return the shared sample schema: `image`, `label`, `meta`.
3. Use stable IDs suitable for caches, manifests, and prediction exports.
4. Fail with actionable messages that name the required env var, expected structure, and an example path.

## Outputs

- Uniform dataset builders and loaders.
- Reproducible split metadata when calibration or held-out subsets are created.

## Success criteria

- External datasets do not assume local availability.
- Loader interfaces stay consistent across train and eval datasets.
