# Data AGENTS.md

Scope: dataset wrappers, transforms, and dataloader builders under `src/owaid/data/`.

Invariants:
- `__getitem__` returns `{"image", "label", "meta"}`.
- Labels use `0=Real`, `1=AI`; real-only datasets still return `label=0` and mark that in metadata.
- Sample IDs must be stable enough for manifests, caches, and prediction export.
- Missing data must fail with actionable env-var and path guidance.

Ownership:
- Dataset-specific parsing lives here.
- Scripts and training code should orchestrate loaders, not duplicate parsing logic.

Split protocol:
- Stratified splitting via sklearn `StratifiedShuffleSplit` with a manual fallback for small or degenerate strata.
- `split_manifest.json` is saved to the `splits/` directory inside the run, including a SHA-256 integrity hash for each split.
- On resume, the manifest is reloaded to guarantee exact reproducibility of train/val/test assignments.

Acceptance:
- Loader interfaces stay uniform across datasets.
- Transforms remain config-driven and deterministic when requested.
