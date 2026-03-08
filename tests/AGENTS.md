# Tests AGENTS.md

Scope: automated tests under `tests/`.

Invariants:
- Keep tests fast, synthetic, and data-free unless a fixture explicitly requires otherwise.
- Prefer narrow unit tests for metrics, calibration, predictors, and artifact loading.
- Do not duplicate implementation details from production code just to make tests pass.

Ownership:
- Behavioral checks belong here.
- Test-only utilities should stay lightweight and local to the test suite.

Acceptance:
- CI-relevant tests run with `pytest -q` without pulling external datasets.
