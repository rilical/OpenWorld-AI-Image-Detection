---
name: baseline-detector
description: Use when implementing or modifying the CLIP baseline detector, residual or DIRE fusion modules, model config knobs, or detector-side inference contracts in this repo. Do not use for dataset parsing, calibration-only changes, or plotting work.
---

# Baseline Detector

## Use when

- Working on `src/owaid/models/clip_detector.py`, fusion modules, or detector config shape.
- Adjusting frozen-backbone behavior, head depth, or optional residual fusion.

## Do not use when

- The task is purely about dataset loading, conformal logic, or evaluation/reporting scripts.

## Workflow

1. Keep the encoder frozen by default unless config explicitly relaxes that.
2. Keep model files free of training-loop and dataset-loading logic.
3. Preserve a clean baseline path when DIRE is disabled.
4. Maintain a stable output contract for training, eval, and inference callers.

## Outputs

- CPU-safe detector modules with explicit config controls.
- Optional fusion components that do not contaminate the baseline path.

## Success criteria

- The model emits two-class logits for `{Real, AI}`.
- Baseline and fusion paths remain separable and testable.
