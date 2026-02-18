# scripts/agents.md

Scope: CLI wrappers around src/owaid modules.

Conventions:
- Every script uses argparse with:
  --config <path>
  --run <run_dir> (for eval scripts that attach to an existing run)
  --device cuda|cpu (optional)
- Scripts must:
  - load config
  - set seed
  - create or use run_dir
  - write artifacts under run_dir per repo schema

Scripts categories:
- training: train_baseline.py, train_with_dire.py
- calibration: calibrate_temperature.py, build_conformal.py
- evaluation: eval_commfor.py, eval_vct2.py, eval_raid.py, eval_aria.py
- plots: make_plots.py
