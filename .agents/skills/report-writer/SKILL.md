---
name: report-writer
description: Use when creating or updating research-facing README content, results summaries, report scaffolding, paper outlines, or figure and table checklists for this project. Do not use for core implementation changes that belong in src/ or scripts/.
---

# Report Writer

## Use when

- Updating repo-level research documentation or report scaffolding.
- Summarizing run metrics into report-friendly text or tables.
- Keeping README and report commands aligned with the actual repo.

## Do not use when

- The task is code-heavy implementation in data, models, training, calibration, or metrics.

## Workflow

1. Pull commands and artifact paths from the live repo, not memory.
2. Keep documentation consistent with current scripts and outputs.
3. Prefer concise structure: motivation, workflow, commands, artifacts, risks.
4. Note empirical-coverage caveats clearly for shifted datasets.

## Outputs

- Updated README or report files.
- Figure and table checklists tied to actual artifacts.

## Success criteria

- Docs do not advertise dead commands.
- Report scaffolding matches the implemented pipeline.
