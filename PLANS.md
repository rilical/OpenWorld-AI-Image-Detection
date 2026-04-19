# PLANS.md

Use this file for tasks that span more than 3 files, alter repo guidance, or change architecture.

## Template

### <task title>
- Goal:
- Files to touch:
- Risks:
- Acceptance checks:

## Active

### 2026-04-16 SGF-Net: Spectral-Gated Forensic Fusion Network

- **Goal**: Implement a novel multi-branch detector (SGF-Net) that fuses CLIP semantic features, spectral forensics (FFT), and pixel-level structural artifacts (NPR + SRM) through a learned spectral gating mechanism. This directly addresses the grader's novelty concern ("What are you doing that others have not done yet?") and the abstention-overuse feedback.
- **Design doc**: `context.md` — full literature review, architecture spec, and ablation plan.
- **Files to create**:
  - `src/owaid/models/spectral_branch.py` — FFT-based spectral forensic encoder
  - `src/owaid/models/pixel_forensic_branch.py` — NPR + SRM pixel artifact encoder
  - `src/owaid/models/gating.py` — spectral-guided attention gating network
  - `src/owaid/models/sgf_net.py` — full SGF-Net assembling all branches + gating + head
  - `src/owaid/training/losses.py` — confidence separation auxiliary loss
  - `configs/sgf_net.yaml` — training config for SGF-Net
  - `scripts/train_sgf.py` — training entry point
- **Files to modify**:
  - `src/owaid/models/__init__.py` — register new model
  - `src/owaid/inference/io.py` — add SGF-Net to `build_model_from_config`
  - `src/owaid/training/train_loop.py` — support pluggable loss function for auxiliary confidence loss
- **Risks**:
  - Spectral/pixel branches add parameters and compute; may need to tune batch size or use AMP
  - Gating network could collapse to uniform weights without careful initialization
  - Confidence separation loss λ needs tuning; too large → underfitting classification, too small → no effect
  - Three-branch model is deeper than baseline; may require more epochs to converge
- **Acceptance checks**:
  - `python3 -m compileall src` passes
  - `pytest -q` passes
  - SGF-Net forward pass works: `model(torch.randn(2,3,224,224))` returns `{"logits": (2,2), "probs": (2,2)}`
  - Gate weights are non-degenerate (not all uniform) after training
  - Training on CommunityForensics-Small completes and produces checkpoints
  - Evaluation on commfor/RAID with all 4 abstention modes produces metrics.json
  - Ablation table (A-F) demonstrates component contributions
  - AUROC on CommunityForensics ≥ 0.97 (baseline is 0.963)
- **Novelty claims** (for final report):
  1. First spectral-gated adaptive fusion of semantic + frequency + pixel forensic branches
  2. First combination of NPR features (CVPR 2024) with CLIP embeddings
  3. Confidence-calibrated auxiliary training loss for forensic detection with abstention
  4. First integrated pipeline: multi-branch forensics → temperature calibration → conformal abstention
- **Implementation phases**:
  - Phase 1: Implement spectral branch, pixel forensic branch, gating network, SGF-Net model class
  - Phase 2: Implement confidence loss, modify training loop, create config + train script
  - Phase 3: Train baseline + SGF-Net + ablations on CommunityForensics
  - Phase 4: Evaluate all models on VCT2, RAID, ARIA (open-world benchmarks)
  - Phase 5: Generate comparison tables (vs. UnivFD, AIDE, NPR, Community Forensics), figures, update report

### 2026-04-16 Final Report Improvements (from milestone feedback)

- **Goal**: Address all grader feedback items for the final report.
- **Files to touch**: Report LaTeX/Markdown, `scripts/make_plots.py`, `assets/figures/system_overview.png`
- **Feedback items**:
  1. ~~Novelty concern~~ → addressed by SGF-Net (see plan above)
  2. Fix system diagram — show 2 output neurons, not 4; clearly separate abstention mechanism from classification
  3. Add precision/recall/F1/confusion matrix to results tables (already computed in eval_loop, just needs to surface in report)
  4. Add quantitative comparison with prior work (UnivFD, AIDE, Community Forensics benchmark numbers)
  5. Show risk-coverage curves with labeled tau values to address abstention-overuse concern
  6. Discuss principled tau selection (target coverage, conformal vs. threshold comparison)
- **Acceptance checks**: Final report addresses each feedback bullet; comparison table includes at least 3 prior methods; confusion matrices included; risk-coverage figure has clear annotations

## Completed

### 2026-03-07 Repo Guidance Migration

- Goal: Align repo instructions with the attached Codex guidance by standardizing on `AGENTS.md`, adding a root planning document, and converting local skills into discoverable `SKILL.md` bundles.
- Files to touch: `AGENTS.md`, nested `*/AGENTS.md`, `.agents/skills/**`, `PLANS.md`.
- Risks: Breaking references from root guidance to nested guidance or local skills; leaving stale lowercase `agents.md` wording after the rename migration.
- Acceptance checks: Root guidance references `AGENTS.md` and `PLANS.md` consistently; skills exist as bundle directories with `SKILL.md` frontmatter; `python -m compileall src`; `pytest -q`.
- Delivered: 12 agents.md files renamed to AGENTS.md, root AGENTS.md trimmed to ~45 lines, routing table + skills pointers, 8 skill bundles created under `.agents/skills/`.

### 2026-03-07 Prompt Pack Audit And Completion

- Goal: Audit the current repo against all 30 milestones in the attached Codex prompt pack, then implement missing or partial pieces so the repo can support training, calibration, evaluation, plotting, reporting, and demo flows from a consistent artifact contract.
- Files to touch: `README.md`, `DATASETS.md`, `Makefile`, `reports/**`, `.github/workflows/**`, `configs/**`, `scripts/**`, `src/owaid/**`, `tests/**`, `notebooks/**`, `demo/**`, `PLANS.md`.
- Risks: Overwriting unrelated user work already in progress; leaving partial interfaces that look complete; introducing duplicated logic between scripts, notebooks, demo, and package modules while trying to close many gaps quickly.
- Acceptance checks: Audit table or summary exists; missing milestone work is implemented in-package instead of duplicated; `python3 -m compileall src`; `pytest -q`; key CLIs at least load with `--help`; docs and artifact layout are consistent with implemented commands.
- Delivered: Unified inference API (`src/owaid/inference/`), threshold abstention baseline, stratified split manifests with SHA-256 hash, bootstrap CIs, empirical conformal coverage, worst-group selective accuracy, publication-quality plotting overhaul, Gradio demo, Makefile with 14 targets, `DATASETS.md`, `reports/` scaffold, `scripts/generate_summary.py`, `scripts/predict_image.py`, `scripts/cache_residuals.py`.
