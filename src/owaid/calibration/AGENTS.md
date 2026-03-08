# Calibration AGENTS.md

Scope: temperature scaling and conformal prediction under `src/owaid/calibration/`.

Invariants:
- Calibration uses held-out logits and labels only; it does not update classifier weights.
- Temperature artifacts live at `outputs/runs/<run_id>/calibration/temperature.json`.
- Conformal artifacts live at `outputs/runs/<run_id>/calibration/conformal.json`.
- Quantile selection must follow the finite-sample split conformal rule.

Ownership:
- Artifact fitting and serialization live here.
- Tri-state mapping logic stays in `src/owaid/models/abstention.py`.

Acceptance:
- Saved artifacts are reusable by eval scripts and any shared inference path.
