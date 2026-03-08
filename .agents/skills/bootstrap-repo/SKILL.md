---
name: bootstrap-repo
description: Use when changing repo-wide scaffolding, AGENTS.md or PLANS.md conventions, artifact layout, make targets, or other shared engineering guidance for this research codebase. Do not use for dataset-, model-, metric-, or experiment-specific implementation work.
---

# Bootstrap Repo

## Use when

- Updating root or nested `AGENTS.md` files.
- Changing repo-wide commands, artifact schema, or scaffold rules.
- Tightening contributor workflow around planning or acceptance checks.

## Do not use when

- The task is isolated to a dataset loader, model, metric, eval script, or demo feature.

## Workflow

1. Read root `AGENTS.md`, then the nearest nested `AGENTS.md` files for touched paths.
2. If the task spans more than 3 files or changes shared workflow, update `PLANS.md` first.
3. Keep root instructions small; move repeatable detail into skills instead of expanding repo-wide guidance.
4. Preserve the artifact contract under `outputs/runs/<run_id>/`.

## Outputs

- Updated repo guidance that is short, consistent, and references the right subtrees.
- Any scaffold or command changes reflected in the root contract.

## Success criteria

- `AGENTS.md` stays concise and operational.
- Shared workflow changes do not contradict nested guidance or actual commands.
