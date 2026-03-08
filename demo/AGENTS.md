# Demo AGENTS.md

Scope: local demo UI under `demo/`.

Interface:
- Gradio Blocks UI launched via `make demo` or `python demo/app.py`.
- Uses the unified `Predictor` from `src/owaid/inference/` for all predictions.
- CLI args: `--run` (run directory), `--device` (cpu/cuda), `--image` (non-interactive single-image fallback).
- UI components: image upload, tri-state label (`AI` / `Real` / `Abstain`), calibrated confidence score, and prediction set display.

Invariants:
- Demo code performs inference only; no training or calibration logic belongs here.
- Predictions should flow through the shared `Predictor` contract instead of bespoke UI-only code paths.
- CPU should work by default; GPU is optional.

Ownership:
- Upload handling, rendering, and presentation belong here.
- Model loading and decision logic belong in package modules.

Acceptance:
- The demo can show tri-state output, calibrated confidence, and prediction-set details from a saved run.
