# Reports

This directory holds paper-facing assets generated from run artifacts.

- `paper.md`: working manuscript draft.
- `paper_outline.md`: section-level paper structure.
- `figures/`: generated figures and figure checklist.
- `tables/`: generated tables and table checklist.
- `results_summary_template.md`: template for run-level result summaries.

Use artifact-backed commands only:

- `python3 scripts/make_plots.py --runs outputs/runs --out reports/figures`
- `python3 scripts/generate_summary.py --run outputs/runs/<run_id> --out reports/results_summary.md`
