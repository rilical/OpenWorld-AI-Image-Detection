# Utils AGENTS.md

Scope: config loading, path helpers, run metadata, logging, and seeding under `src/owaid/utils/`.

Invariants:
- Run directory creation and artifact writes stay explicit.
- CLI entrypoints should route config, seed, and metadata setup through this layer.
- Missing env vars and bad paths must fail with actionable messages.
- Keep abstractions minimal; do not introduce framework-like config systems.

Ownership:
- Shared utility functions belong here.
- Domain-specific dataset, model, or metric logic does not.

Acceptance:
- Scripts can build a run directory, write `config.yaml`, and write `meta.json` without duplicating helper code.
