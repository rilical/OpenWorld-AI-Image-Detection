# notebooks/agents.md

Scope: research notebooks used for exploration and figure generation.

Rules:
- Notebooks should call scripts or import src/owaid modules; do not duplicate logic.
- Assume outputs already exist under outputs/runs.
- Each notebook should have:
  - a "Setup" cell (paths, imports)
  - a "Load runs" cell
  - a "Generate figures" section matching report figures

Expected notebooks:
- 01_data_exploration.ipynb
- 02_train_baseline.ipynb
- 03_calibration_and_abstention.ipynb
- 04_open_world_evals.ipynb
