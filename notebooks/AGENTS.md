# Notebooks AGENTS.md

Scope: research notebooks in `notebooks/`.

Invariants:
- Notebooks consume run artifacts or stable library APIs; they do not copy training or eval loops.
- Missing outputs should fail gracefully with setup guidance.
- Notebook structure should stay report-oriented, not ad hoc scratchpad-heavy.

Ownership:
- Exploration, analysis, and artifact-backed storytelling belong here.
- Reusable logic belongs in the package or scripts.

Acceptance:
- Notebook cells can be rerun from repo root with minimal manual patching.
