# Metrics AGENTS.md

Scope: numeric metrics and plot-ready curve builders under `src/owaid/metrics/`.

Invariants:
- Binary labels are `0=Real`, `1=AI`.
- `y_score` means the class-1 or AI score.
- Single-class edge cases must return informative results instead of crashing.
- Keep APIs numpy-friendly so scripts and tests can call them directly.

Ownership:
- Metric math lives here.
- Training, eval, and plotting code should consume these functions rather than reimplementing them.

Key additions:
- `selective.py`:
  - `bootstrap_ci(metric_fn, *args, n_bootstrap=1000, confidence=0.95)` — bootstrap confidence intervals for any metric function.
  - `worst_group_selective_accuracy(correct_mask, answered_mask, group_ids)` — selective accuracy for the worst-performing group.
- `calibration_metrics.py`:
  - `empirical_conformal_coverage(prediction_sets, labels, group_ids=None)` — empirical coverage of conformal prediction sets, optionally per group.

Acceptance:
- AUROC, TPR@1%FPR, ECE, risk-coverage, AURC, abstain rate, and selective accuracy remain available.
