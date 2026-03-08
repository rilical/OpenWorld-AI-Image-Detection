# PLANS.md

Use this file for tasks that span more than 3 files, alter repo guidance, or change architecture.

## Template

### <task title>
- Goal:
- Files to touch:
- Risks:
- Acceptance checks:

## Active

(none)

## Completed

### 2026-03-07 Repo Guidance Migration

- Goal: Align repo instructions with the attached Codex guidance by standardizing on `AGENTS.md`, adding a root planning document, and converting local skills into discoverable `SKILL.md` bundles.
- Files to touch: `AGENTS.md`, nested `*/AGENTS.md`, `.agents/skills/**`, `PLANS.md`.
- Risks: Breaking references from root guidance to nested guidance or local skills; leaving stale lowercase `agents.md` wording after the rename migration.
- Acceptance checks: Root guidance references `AGENTS.md` and `PLANS.md` consistently; skills exist as bundle directories with `SKILL.md` frontmatter; `python -m compileall src`; `pytest -q`.
- Delivered: 12 agents.md files renamed to AGENTS.md, root AGENTS.md trimmed to ~45 lines, routing table + skills pointers, 8 skill bundles created under `.agents/skills/`.

### 2026-03-07 Prompt Pack Audit And Completion

- Goal: Audit the current repo against all 30 milestones in the attached Codex prompt pack, then implement missing or partial pieces so the repo can support training, calibration, evaluation, plotting, reporting, and demo flows from a consistent artifact contract.
- Files to touch: `README.md`, `DATASETS.md`, `Makefile`, `reports/**`, `.github/workflows/**`, `configs/**`, `scripts/**`, `src/owaid/**`, `tests/**`, `notebooks/**`, `demo/**`, `PLANS.md`.
- Risks: Overwriting unrelated user work already in progress; leaving partial interfaces that look complete; introducing duplicated logic between scripts, notebooks, demo, and package modules while trying to close many gaps quickly.
- Acceptance checks: Audit table or summary exists; missing milestone work is implemented in-package instead of duplicated; `python3 -m compileall src`; `pytest -q`; key CLIs at least load with `--help`; docs and artifact layout are consistent with implemented commands.
- Delivered: Unified inference API (`src/owaid/inference/`), threshold abstention baseline, stratified split manifests with SHA-256 hash, bootstrap CIs, empirical conformal coverage, worst-group selective accuracy, publication-quality plotting overhaul, Gradio demo, Makefile with 14 targets, `DATASETS.md`, `reports/` scaffold, `scripts/generate_summary.py`, `scripts/predict_image.py`, `scripts/cache_residuals.py`.
