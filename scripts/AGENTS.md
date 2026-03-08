# Scripts AGENTS.md

Scope: thin CLI wrappers in `scripts/`.

Invariants:
- Scripts orchestrate library code; they do not duplicate core logic from `src/owaid/`.
- Every script must run from repo root as `python scripts/<name>.py ...`.
- Training and calibration scripts create or extend run artifacts under `outputs/runs/<run_id>/`.

Ownership:
- Argument parsing, config loading, and high-level orchestration belong here.
- Shared business logic belongs in the package.

New scripts:
- `generate_summary.py` — generates a markdown results summary from eval metrics.
- `predict_image.py` — single-image CLI inference using the unified `Predictor`.
- `cache_residuals.py` — offline DIRE residual caching for faster training/eval.

CLI flags:
- Eval scripts now support `--abstention-method` (`conformal` or `threshold`) and `--tau` flags.

Acceptance:
- Entrypoints remain thin, explicit, and compatible with the repo artifact schema.
