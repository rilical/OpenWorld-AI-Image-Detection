# agents.md — Open‑World AI Image Detection with Abstention (Repo Map Head)

This file is the single source of truth for agent context. Any coding agent should:
1) Read THIS file first (repo contract + routing map).
2) Then jump to the relevant module-level agents.md file(s) listed below.
3) Only then open the specific code files needed for the task.

Goal: keep context tight and avoid token waste. Load the smallest slice that lets you do correct work.

--------------------------------------------------------------------------------
PROJECT CONTRACT (NON‑NEGOTIABLES)
--------------------------------------------------------------------------------

We are building an open‑world AI‑image detector that outputs: {AI, Real, Abstain}.

1) Training data (core):
   - Train/val on Hugging Face dataset: OwensLab/CommunityForensics-Small
   - Create a held‑out calibration split from CommunityForensics-Small (NOT used for training).

2) Model (baseline):
   - Pretrained vision encoder (OpenCLIP/CLIP ViT or similar)
   - Frozen feature extractor by default
   - Small MLP head for binary classification: {AI, Real}
   - Optional partial fine‑tune mode (config-controlled) allowed, but default is frozen.

3) Optional forensic branch (DIRE-style):
   - Reconstruct x with pretrained diffusion model -> x_hat
   - Residual map: R = |x − x_hat|
   - Encode R and fuse with CLIP features (concat + MLP is the default).

4) Abstention layer (core):
   - Temperature scaling on calibration set
   - Split conformal prediction for classification to produce prediction set Γ(x) over {AI, Real}
   - Decision:
       Γ(x)={AI}   -> AI
       Γ(x)={Real} -> Real
       Γ(x)={AI,Real} or empty -> Abstain

5) Open‑world evaluation targets:
   - VCT² benchmark (unseen generators)
   - RAID (transferable adversarial examples)
   - ARIA (real-only dataset for false positives)

6) Metrics that MUST exist in code:
   - AUROC
   - TPR @ 1% FPR
   - Expected Calibration Error (ECE)
   - Risk–coverage curves + AURC (area under risk–coverage)
   - Abstention rate
   - Accuracy on non‑abstained subset (selective accuracy)

--------------------------------------------------------------------------------
REPO INVARIANTS (DO NOT BREAK)
--------------------------------------------------------------------------------

Package:
- Python package lives in: src/owaid
- Scripts live in: scripts/
- Configs live in: configs/
- Outputs live in: outputs/runs/<run_id>/

Artifact schema (every run):
outputs/runs/<run_id>/
  config.yaml                 (resolved config used)
  meta.json                   (git hash, timestamp, host, torch versions)
  checkpoints/
    best.pt
    last.pt
  calibration/
    temperature.json
    conformal.json
  eval/
    <dataset_name>/
      metrics.json
      predictions.parquet     (optional but strongly recommended)
  logs/
    train.jsonl               (one JSON per step/epoch)
    eval.jsonl

Code style:
- Prefer explicit, minimal abstractions.
- No hidden “magic registries” unless it simplifies code and is documented.
- All public functions must have docstrings and typed inputs/outputs.
- Every CLI script must be runnable from repo root: `python scripts/<name>.py ...`

Data access:
- Never assume data exists; fail with actionable message:
  - required env var (e.g., VCT2_ROOT, ARIA_ROOT)
  - expected directory structure
  - example path
- CommunityForensics-Small loads through Hugging Face `datasets` API.

--------------------------------------------------------------------------------
ROUTING MAP (NEXT CONTEXT HOPS)
--------------------------------------------------------------------------------

If you are working on…

Data + transforms:
- read: src/owaid/data/agents.md

Models (CLIP baseline, fusion, DIRE):
- read: src/owaid/models/agents.md

Training/eval loops + checkpointing:
- read: src/owaid/training/agents.md

Calibration + conformal prediction:
- read: src/owaid/calibration/agents.md

Metrics (AUROC/TPR@FPR, ECE, selective risk-coverage/AURC):
- read: src/owaid/metrics/agents.md

Utilities (config, logging, seed, paths):
- read: src/owaid/utils/agents.md

CLI scripts:
- read: scripts/agents.md

Demo UI:
- read: demo/agents.md

Experiment configs:
- read: configs/agents.md

Notebooks:
- read: notebooks/agents.md

Tests:
- read: tests/agents.md

--------------------------------------------------------------------------------
INJECTION POINTS (WHERE YOU EXTEND WITHOUT REFACTORING)
--------------------------------------------------------------------------------

New dataset:
- Create loader in src/owaid/data/<dataset>.py
- Add builder function in src/owaid/data/__init__.py
- Add config in configs/eval_<dataset>.yaml
- Add eval script in scripts/eval_<dataset>.py (thin wrapper)

New backbone:
- Extend src/owaid/models/clip_detector.py to accept open_clip model name + pretrained tag via config.
- Do NOT leak backbone-specific logic into training loop.

New abstention method:
- Implement in src/owaid/models/abstention.py and/or src/owaid/calibration/
- Training loop stays unchanged; eval loop calls abstention policy.

New corruption/robustness transform:
- Add to src/owaid/data/transforms.py
- Toggle via config (never hardcode)

New plots:
- Implement in scripts/make_plots.py
- Use only artifacts in outputs/runs/ (don’t depend on “live” objects)

--------------------------------------------------------------------------------
DEFAULTS (SAFE CHOICES)
--------------------------------------------------------------------------------

Backbone default:
- OpenCLIP ViT-B/32 or ViT-L/14 (config-controlled; start small).

Classifier head default:
- 2-layer MLP with dropout, output logits [B,2].

Temperature scaling:
- Fit one scalar temperature T by minimizing NLL on calibration set.

Conformal default:
- Split conformal classification with nonconformity s(x,y)=1-p_T(y|x).
- Provide optional Mondrian (class-conditional) conformal.

Decision default:
- Conformal prediction set -> tri-state output as specified.

--------------------------------------------------------------------------------
"NO DRAMA" CHECKLIST
--------------------------------------------------------------------------------

Before you finalize any change:
- Run at least: `python -m compileall src`
- Run unit tests if present: `pytest -q`
- Confirm script entrypoints still work from repo root.
- Confirm outputs write to outputs/runs/<run_id>/ with schema above.

If you’re unsure where to edit:
- Stop. Read the relevant module agents.md.
