# tests/agents.md

Scope: unit tests for metrics + calibration.

Rules:
- Avoid heavyweight dataset downloads in CI.
- Use small synthetic tensors for metrics/conformal tests.
- Provide at least:
  - ECE sanity test (perfect calibration -> low ECE)
  - Conformal quantile selection test (monotonicity + coverage on synthetic)
  - TPR@FPR test on constructed scores
