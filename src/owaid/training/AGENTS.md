# Training AGENTS.md

Scope: training loops, evaluation loops, and checkpointing under `src/owaid/training/`.

Invariants:
- Keep loops explicit and minimal; no framework-heavy orchestration.
- JSONL logging and checkpoint writing must use the shared utils layer.
- Evaluation stays separate from training and is the place where abstention policies are compared.

Ownership:
- Optimization, checkpoint lifecycle, and metric aggregation belong here.
- Model architecture and calibration math belong in sibling modules.

Eval loop contract:
- `abstention_method` parameter: `"conformal"` or `"threshold"` selects the abstention policy applied during evaluation.
- `tau` parameter: threshold value used when `abstention_method="threshold"`.
- New metrics wired in: bootstrap CIs (via `bootstrap_ci`), empirical conformal coverage (via `empirical_conformal_coverage`), and worst-group selective accuracy (via `worst_group_selective_accuracy`).

Acceptance:
- `best.pt` and `last.pt` remain compatible with the rest of the repo.
- Eval outputs contain enough data to write metrics and prediction exports without UI-only code paths.
