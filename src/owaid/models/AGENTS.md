# Models AGENTS.md

Scope: detector models, fusion modules, and abstention wrappers under `src/owaid/models/`.

Invariants:
- Model code stays free of training-loop and dataset-loading logic.
- The CLIP encoder is frozen by default and configurable through model config.
- DIRE or residual fusion remains optional and must not change the baseline path when disabled.
- Abstention uses saved calibration artifacts and does not retrain classifier weights.

Ownership:
- Backbone, head, residual encoder, and fusion code belong here.
- Training behavior belongs in `src/owaid/training/`.

Key additions in `abstention.py`:
- `predict_with_threshold_abstention(logits, temperature, tau)` — apply temperature scaling and abstain when max softmax probability is below `tau`.
- `sweep_tau(logits, labels, temperature, n_steps)` — sweep threshold values to find the best tau on a validation set.

Acceptance:
- Forward paths stay CPU-safe.
- Output contracts remain consistent for training, eval, and inference consumers.
