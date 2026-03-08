# Inference AGENTS.md

Scope: shared prediction and artifact-loading code under `src/owaid/inference/`.

Invariants:
- Inference code is the single contract used by eval scripts, CLIs, and the demo.
- Run loading, checkpoint loading, temperature handling, and conformal handling stay here instead of being reimplemented in scripts.
- Results should be normalized enough that UI and evaluation code can consume them consistently.

Ownership:
- Predictor wrappers, run-artifact IO, and result normalization belong here.
- Model architecture, calibration fitting, and plotting do not.

Key exports:
- `Predictor` class — unified prediction interface used by eval, CLI, and demo.
- `load_run()` factory — loads a complete run (model + calibration) from a run directory.
- `io.py` helpers:
  - `build_model_from_config` — construct model from a config dict.
  - `load_calibration_artifacts` — load temperature and conformal artifacts.
  - `load_checkpoint` — load model weights from a checkpoint file.
  - `load_run_config` — load `config.yaml` from a run directory.
  - `resolve_checkpoint_path` — resolve `best.pt` / `last.pt` from a run directory.

Acceptance:
- A saved run can be loaded once and used consistently from CLI, eval, and demo entrypoints.
