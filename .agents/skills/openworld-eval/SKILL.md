---
name: openworld-eval
description: Use when building or changing evaluation scripts, prediction export, dataset-specific benchmarking, selective metrics, or artifact-backed evaluation flows for CommunityForensics-Small, VCT2, RAID, or ARIA. Do not use for training-loop internals or repo scaffolding.
---

# Open-World Eval

## Use when

- Adding or modifying `scripts/eval_*.py`.
- Changing dataset-level evaluation modes or prediction export.
- Reporting selective metrics, empirical coverage, or dataset-specific stress results.

## Do not use when

- The task is only about model construction, repo guidance, or plot styling.

## Workflow

1. Load saved checkpoints and calibration artifacts from a run directory.
2. Reuse package metric functions instead of recomputing metrics in scripts.
3. Export per-sample predictions with stable IDs and dataset metadata.
4. Handle real-only datasets and single-class edge cases explicitly.

## Outputs

- Saved `metrics.json` and prediction exports under `eval/<dataset>/`.
- Side-by-side comparisons for forced, threshold, and conformal decisions when applicable.

## Success criteria

- Eval scripts share a uniform contract.
- Missing datasets fail cleanly with setup guidance instead of stack traces.
