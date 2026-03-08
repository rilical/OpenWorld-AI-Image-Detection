---
name: demo-ui
description: Use when building or changing the local demo UI for tri-state image prediction, run selection, calibrated confidence display, or residual visualization backed by saved run artifacts. Do not use for model training, dataset ingestion, or report generation.
---

# Demo UI

## Use when

- Implementing or adjusting `demo/app.py`.
- Adding run selection, image upload, prediction display, or residual visualization.
- Wiring the UI to shared inference code and saved calibration artifacts.

## Do not use when

- The task is about training loops, dataset loading, or publication reporting.

## Workflow

1. Route all predictions through shared package inference code.
2. Keep the demo inference-only and artifact-backed.
3. Support CPU by default and enable GPU opportunistically.
4. Present tri-state label, confidence, prediction set, and abstention state clearly.

## Outputs

- A local demo that can load a saved run and score uploaded images.

## Success criteria

- The UI does not duplicate model-loading or decision logic.
- Demo behavior matches CLI and eval behavior for the same run artifacts.
