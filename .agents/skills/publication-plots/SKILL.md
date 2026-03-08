---
name: publication-plots
description: Use when generating artifact-backed figures or tables such as risk-coverage curves, reliability diagrams, metric summaries, or per-generator breakdowns from outputs/runs data. Do not use for live training analysis or model implementation work.
---

# Publication Plots

## Use when

- Implementing or adjusting `scripts/make_plots.py`.
- Building figure-ready CSV tables from saved run artifacts.
- Producing report figures from `metrics.json` or prediction exports.

## Do not use when

- The task is about model logic, dataset loading, or training execution.

## Workflow

1. Read only saved run artifacts, never live in-memory training objects.
2. Keep plots reproducible from `outputs/runs/`.
3. Prefer `matplotlib` unless a stronger dependency is justified.
4. Export the data behind each figure alongside the rendered assets.

## Outputs

- Plot files under the configured output location.
- Plot-ready CSV or JSON tables for reports.

## Success criteria

- Figures can be regenerated from artifacts alone.
- Plot data stays aligned with the metrics contract used by eval scripts.
