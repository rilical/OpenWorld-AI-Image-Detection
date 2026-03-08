# AGENTS.md - Open-World AI Image Detection with Abstention

Open-world AI-image detector returning `{AI, Real, Abstain}` with a CLIP baseline, temperature scaling, and split conformal prediction. External evaluation targets are VCT2, RAID, and ARIA.

## Before Editing

- Read this file first, then the nearest nested `AGENTS.md` for every path you will touch.
- If the task spans more than 3 files, changes architecture, or adds dependencies, create or update a section in `PLANS.md` first with goal, files, risks, and acceptance checks.
- If a matching skill exists under `.agents/skills/`, use it instead of expanding a long inline workflow.
- Do not scan unrelated directories.

## Commands

- `make smoke` — compileall + pytest
- `make test` — pytest only
- `make plots` — publication figures to `reports/figures/`
- `make demo` — launch Gradio UI
- `make summary` — generate results summary
- `python3 -m compileall src`
- `python3 -m pytest -q`

## Repo Layout

- Package: `src/owaid/`
- Scripts: `scripts/`
- Configs: `configs/`
- Outputs: `outputs/runs/<run_id>/`

## Artifact Layout

```text
outputs/runs/<run_id>/
  config.yaml
  meta.json
  checkpoints/{best.pt,last.pt}
  calibration/{temperature.json,conformal.json}
  splits/split_manifest.json
  eval/<dataset>/metrics.json
  logs/{train.jsonl,eval.jsonl}
```

## Routing

| Working on... | Read |
|---|---|
| Data + transforms | `src/owaid/data/AGENTS.md` |
| Models | `src/owaid/models/AGENTS.md` |
| Shared inference | `src/owaid/inference/AGENTS.md` |
| Training or evaluation loops | `src/owaid/training/AGENTS.md` |
| Calibration or conformal | `src/owaid/calibration/AGENTS.md` |
| Metrics | `src/owaid/metrics/AGENTS.md` |
| Utilities | `src/owaid/utils/AGENTS.md` |
| CLI scripts | `scripts/AGENTS.md` |
| Demo UI | `demo/AGENTS.md` |
| Configs | `configs/AGENTS.md` |
| Notebooks | `notebooks/AGENTS.md` |
| Tests | `tests/AGENTS.md` |

## Skills

| Skill | Use when |
|---|---|
| `.agents/skills/bootstrap-repo/` | Repo-wide guidance, artifact layout, make targets, scaffold policy |
| `.agents/skills/dataset-loader/` | Dataset contracts, split manifests, env-var based loading |
| `.agents/skills/baseline-detector/` | CLIP baseline or DIRE fusion model work |
| `.agents/skills/calibration-conformal/` | Temperature scaling, conformal, abstention policy changes |
| `.agents/skills/openworld-eval/` | Eval scripts, prediction export, dataset-level metrics |
| `.agents/skills/publication-plots/` | Artifact-only plotting and tables |
| `.agents/skills/report-writer/` | Results summaries and report scaffolding |
| `.agents/skills/demo-ui/` | Local tri-state inference demo work |

## After Editing

- Run the narrowest relevant check, at minimum `python -m compileall src`.
- Run `pytest -q` when tests apply.
- Report changed files, what passed, and any remaining risks.
