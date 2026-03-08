---
name: calibration-conformal
description: Use when implementing or adjusting temperature scaling, split or Mondrian conformal prediction, abstention policies, or calibration artifacts for this detector. Do not use for model architecture, dataset ingestion, or plotting work.
---

# Calibration Conformal

## Use when

- Changing temperature fitting or artifact serialization.
- Implementing conformal thresholds, prediction sets, or tri-state mapping.
- Comparing forced-decision, threshold-abstention, and conformal-abstention behavior.

## Do not use when

- The task is limited to model backbones, dataset loading, or paper/report output.

## Workflow

1. Fit calibration artifacts on held-out logits and labels only.
2. Use calibrated probabilities for conformal nonconformity scores.
3. Apply the finite-sample quantile rule for split conformal.
4. Keep reusable decision logic separate from training loops.

## Outputs

- `temperature.json` and `conformal.json` artifacts.
- Reusable abstention functions for forced, threshold, and conformal decisions.

## Success criteria

- Classifier weights are unchanged by calibration.
- Artifacts can be reused by eval and shared inference code.
