# Configs AGENTS.md

Scope: YAML experiment configs in `configs/`.

Invariants:
- Configs should describe behavior, not hide it behind implicit defaults scattered across code.
- Keep top-level structure stable enough that scripts and utilities can share parsing logic.
- New config files should extend the existing schema rather than inventing one-off keys.

Ownership:
- Dataset, model, train, calibration, eval, and output knobs belong here.
- Code should validate or document config expectations rather than duplicating config files.

Acceptance:
- Configs remain readable, composable, and aligned with actual script entrypoints.
