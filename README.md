# Open‑World AI Image Detection with Abstention (CMU ECE 18‑786)

**Repository**: 'OpenWorld-AI-Image-Detection'

**Authors**
- Omar Ghabayen — oghabaye@andrew.cmu.edu
- Gorkem Yar — gyar@andrew.cmu.edu
- Abdullah Almansour — aalmanso@andrew.cmu.edu
- Zongchi Xie — zongchix@andrew.cmu.edu
- Youlin Qu — ynq@andrew.cmu.edu

## Abstract

Binary AI-vs-Real detectors often report high accuracy while failing when inputs move outside training conditions (unseen generators, post-processing pipelines, adversarial transfer, and real-world distribution shifts). This repo implements a practical open-world detector that returns three outcomes: **AI**, **Real**, or **Abstain**.

The system is organized as a three-stage reliability pipeline:
1) **train** a frozen-encoder CLIP classifier,
2) **calibrate** confidence on a held-out split,
3) apply **abstention policies** (temperature scaling, confidence-threshold baseline, split conformal prediction) and report selective-risk behavior under out-of-domain benchmarks.

A paper-style figure is placed at `assets/figures/system_overview.png` and referenced below (insert your final diagram artifact):

<p align="center">
  <img src="assets/figures/system_overview.png" width="900" alt="System overview diagram">
</p>
<p align="center">
  <em><b>Figure 1.</b> Open‑world AI image detection architecture. A CLIP encoder produces image embeddings for a binary AI/Real head. Optionally, a DIRE-style residual branch fuses reconstruction residual features with CLIP features. Calibration stage produces a temperature and conformal thresholds on a held-out split. Inference returns a prediction set over {AI, Real}; singleton sets map to labels, non-singletons/empty map to Abstain.</em>
</p>

## Research framing and scope

This project is designed as a reliability-first detector benchmark with explicit failure handling:

- Open-world benchmarks are intentionally mismatched to train data.
- Confidence is not treated as truth; it is **calibrated and managed**.
- Abstention is explicit and measured via risk–coverage.
- Results are evaluated with both **point metrics** and **selective metrics**.

## Why this repo exists

Standard detector pipelines often force binary decisions, which hides uncertainty. In realistic deployment, abstention is often the correct operational action when uncertainty or domain shift is high. This repo provides a reusable scaffold that makes abstention first-class and reproducible.

## Quick links

- Dataset setup: `DATASETS.md`
- Report assets: `reports/`
- Agent guidance: `AGENTS.md`

## Key contributions and ablations

1. **Baseline detector**
   - Frozen OpenCLIP/CLIP-like vision encoder
   - Small MLP head for binary classification (AI/Real)

2. **Calibration + abstention layer**
   - Temperature scaling on calibration split
   - Split conformal prediction over {AI, Real}
   - Optional class-conditional (Mondrian) conformal mode

3. **Abstention policy baselines**
   - Forced decision (no abstention)
   - Temperature-only decisions
   - **Confidence-threshold abstention** (mandatory baseline)
     - abstain when `max_softmax < τ`
     - τ tuned on calibration set for target coverage
   - Conformal abstention

4. **Open-world eval**
   - VCT², RAID, ARIA

5. **Optional DIRE path**
   - Residual branch for image forensic signal
   - Cache-first design for practical compute management

## Data and reality checks (important)

> See [DATASETS.md](DATASETS.md) for full dataset acquisition instructions, env vars, and folder structures.

### CommunityForensics-Small (CFS)

- Source: https://huggingface.co/datasets/OwensLab/CommunityForensics-Small
- Scale: ~278K real + ~278K AI samples
- License: CC BY-NC-SA (important for non-commercial constraints)
- Use split strategy:
  - `train`: classification training
  - Held-out calibration subset from train (default fraction from config)
  - optional validation via dataset split
- Data size is large; configure cache and local path carefully:
  - `HF_HOME`
  - `HF_DATASETS_CACHE`

### VCT²

- Benchmark repository: https://github.com/nasrinimapour/VCT2
- In practice, public resources may be consumed via HF prompt/image tables:
  - https://huggingface.co/datasets/anonymous1233/COCO_AII
  - https://huggingface.co/datasets/anonymous1233/twitter_AII
- Approx size: around 130K images depending on selected subset.

### RAID

- HF dataset: https://huggingface.co/datasets/aimagelab/RAID
- Paper scale: ~72K adversarial examples
- Repo: https://github.com/pralab/RAID

### ARIA (this project)

- ARIA here refers to AdvAIArtProject’s “Adversarial AI‑Art” data, not Meta’s ARIA corpus.
- Source: https://github.com/AdvAIArtProject/AdvAIArt (download path linked in repository documentation)
- Real-only benchmark for false-positive behavior and domain-shift stress.

### Distribution shift caveat (explicit)

Split conformal guarantees assume calibration and test data are exchangeable/i.i.d. This is not true for open-world sets (VCT²/RAID/ARIA) by construction. We therefore report:

- empirical coverage (overall and by class where possible)
- risk–coverage behavior under shift
- clear caveat that out-of-domain coverage is empirical, not distributional-guaranteed

## Optional DIRE implementation policy

DIRE-style residual reconstruction can be implementation-heavy. This scaffold keeps it optional and safe:

- residual feature extraction separated into offline cache workflow
- train/eval can proceed without residual branch
- if enabled, reuse cached residuals for train/val/cal/test to reduce repeated recomputation
- if full diffusion reconstruction is unavailable, use lightweight placeholder (e.g., autoencoder-like residual branch) with explicit ablation note.

A practical starting recipe:

1) extract residual embeddings/images offline
2) cache to disk with stable sample IDs
3) train fusion head on cached residual features

## Repository layout

```text
OpenWorld-AI-Image-Detection/
  README.md
  requirements.txt
  Makefile
  PLANS.md
  DATASETS.md
  IMPLEMENTATION_CHECKLIST.md
  AGENTS.md
  .gitignore
  LICENSE
  CITATION.cff

  .agents/
    skills/
      bootstrap-repo/SKILL.md
      dataset-loader/SKILL.md
      baseline-detector/SKILL.md
      calibration-conformal/SKILL.md
      openworld-eval/SKILL.md
      publication-plots/SKILL.md
      report-writer/SKILL.md
      demo-ui/SKILL.md

  assets/
    figures/
      system_overview.png

  configs/
    baseline_clip.yaml
    dire_fusion.yaml
    calibration.yaml
    eval_commfor.yaml
    eval_vct2.yaml
    eval_raid.yaml
    eval_aria.yaml

  src/
    owaid/
      __init__.py

      data/
        __init__.py
        commfor_small.py
        vct2.py
        raid.py
        aria.py
        transforms.py

      models/
        __init__.py
        clip_detector.py
        dire_residual.py
        fusion.py
        abstention.py

      training/
        __init__.py
        train_loop.py
        eval_loop.py
        checkpoints.py

      calibration/
        __init__.py
        temperature_scaling.py
        conformal.py

      metrics/
        __init__.py
        classification.py
        calibration_metrics.py
        selective.py
        bootstrap.py

      inference/
        __init__.py
        io.py
        predictor.py

      utils/
        __init__.py
        config.py
        logging.py
        seed.py
        paths.py
        run.py

  scripts/
    train_baseline.py
    train_with_dire.py
    calibrate_temperature.py
    build_conformal.py
    evaluate.py
    make_plots.py
    generate_summary.py
    predict_image.py
    cache_residuals.py

  notebooks/
    01_data_exploration.ipynb
    02_train_baseline.ipynb
    03_calibration_and_abstention.ipynb
    04_open_world_evals.ipynb

  demo/
    app.py

  outputs/
    runs/
      .gitkeep
    plots/
      .gitkeep

  reports/
    figures/
      .gitkeep
    tables/
      .gitkeep
    paper.md

  tests/
    test_metrics.py
    test_conformal.py

  .github/
    workflows/
      ci.yml
```

## Quick bootstrap

```bash
# Create base repository structure
mkdir -p assets/figures assets/examples/example_images \
  configs src/owaid/{data,models,training,calibration,metrics,inference,utils} \
  scripts notebooks demo outputs/runs outputs/plots reports/{figures,tables} \
  tests .github/workflows .agents/skills

# Initialize package
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Copy this scaffold into git
git init
git branch -M main

# Verify everything compiles and tests pass
make smoke
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Install PyTorch for your platform from https://pytorch.org/get-started/locally/.

If using large datasets, point caches to dedicated disk:

```bash
export HF_HOME=/path/to/big_disk/hf
export HF_DATASETS_CACHE=/path/to/big_disk/hf/datasets
```

## Smoke checks

```bash
make smoke
pytest -q
```

## Configs and runtime contract

All runs are config-driven YAMLs in `configs/`.

Core schema (high level):
- `seed`, `deterministic`, `device`
- `output.root`, `output.run_name`
- `data.batch_size`, `num_workers`, `img_size`, `data.transforms`
- `model.type` and `model.clip` (OpenCLIP params), `model.head`
- `train` hyper-params and checkpoint metric
- `calibration` method flags
- `eval.save_predictions`

## Pipeline (recommended order)

1. **Train baseline**
```bash
make train-baseline
```

2. **Fit temperature on calibration split**
```bash
make calibrate RUN=outputs/runs/<run_id>
```

3. **Build conformal thresholds**
```bash
make conformal RUN=outputs/runs/<run_id>
```

4. **Evaluate with the four comparison modes**
```bash
make eval-commfor RUN=outputs/runs/<run_id>
make eval-vct2   RUN=outputs/runs/<run_id>
make eval-raid   RUN=outputs/runs/<run_id>
make eval-aria   RUN=outputs/runs/<run_id>
```

5. **Plot reliability and risk-coverage**
```bash
make plots   RUN=outputs/runs/<run_id>
make summary RUN=outputs/runs/<run_id>
```

6. **Optional DIRE cache + training path**
```bash
make cache-residuals RUN=outputs/runs/<run_id>
python3 scripts/train_with_dire.py --config configs/dire_fusion.yaml
```

7. **Local prediction and demo**
```bash
make predict RUN=outputs/runs/<run_id> IMAGE=path/to/image.png
make demo    RUN=outputs/runs/<run_id>
```

<details>
<summary>Raw commands (without Make)</summary>

```bash
# 1. Train baseline
python3 scripts/train_baseline.py --config configs/baseline_clip.yaml

# 2. Fit temperature
python3 scripts/calibrate_temperature.py \
  --config configs/calibration.yaml \
  --ckpt outputs/runs/<run_id>/checkpoints/best.pt \
  --run outputs/runs/<run_id>

# 3. Build conformal thresholds
python3 scripts/build_conformal.py \
  --config configs/calibration.yaml \
  --ckpt outputs/runs/<run_id>/checkpoints/best.pt \
  --run outputs/runs/<run_id> \
  --temperature outputs/runs/<run_id>/calibration/temperature.json

# 4. Evaluate
python3 scripts/evaluate.py --dataset commfor --config configs/eval_commfor.yaml --run outputs/runs/<run_id> --evaluation-mode all
python3 scripts/evaluate.py --dataset vct2 --config configs/eval_vct2.yaml --run outputs/runs/<run_id> --evaluation-mode all
python3 scripts/evaluate.py --dataset raid --config configs/eval_raid.yaml --run outputs/runs/<run_id> --evaluation-mode all
python3 scripts/evaluate.py --dataset aria --config configs/eval_aria.yaml --run outputs/runs/<run_id> --evaluation-mode all

# 5. Plots and summary
python3 scripts/make_plots.py --runs outputs/runs --out reports/figures --style publication
python3 scripts/generate_summary.py --run outputs/runs/<run_id> --out reports/results_summary.md

# 6. DIRE cache + training
python3 scripts/cache_residuals.py --config configs/dire_fusion.yaml --run outputs/runs/<run_id>
python3 scripts/train_with_dire.py --config configs/dire_fusion.yaml

# 7. Predict and demo
python3 scripts/predict_image.py --run outputs/runs/<run_id> --image path/to/image.png
python3 demo/app.py --run outputs/runs/<run_id>
```

</details>

## Makefile targets

```
make smoke          # compileall + pytest
make test           # pytest -q
make train-baseline # train CLIP detector
make calibrate      # fit temperature scaling
make conformal      # build conformal thresholds
make eval-commfor   # evaluate on CommunityForensics
make eval-vct2      # evaluate on VCT2
make eval-raid      # evaluate on RAID
make eval-aria      # evaluate on ARIA
make plots          # publication-quality figures → reports/figures/
make demo           # launch Gradio UI
make summary        # generate reports/results_summary.md
make predict        # single-image inference
make cache-residuals # cache DIRE residuals offline
```

> Configurable via `RUN=outputs/runs/<run_id>`, `CONFIG=configs/baseline_clip.yaml`, `DEVICE=cpu|cuda`.

## Interactive demo

The Gradio Blocks interface accepts an image upload and returns a tri-state label (AI/Real/Abstain), confidence score, and prediction set. Launch with:

```bash
make demo RUN=outputs/runs/<run_id>
# or directly:
python3 demo/app.py --run outputs/runs/<run_id> --device cpu
```

For non-interactive single-image inference:

```bash
python3 demo/app.py --run outputs/runs/<run_id> --image path/to/image.jpg
```

## Evaluation modes

Every evaluation script can compare four decision modes:

1. `forced`: raw argmax without temperature scaling.
2. `temperature`: argmax after temperature scaling.
3. `threshold`: abstain when max calibrated confidence is below `tau`.
4. `conformal`: split conformal prediction sets mapped to `{AI, Real, Abstain}`.

Open-world empirical coverage on VCT2, RAID, and ARIA should be read as an observed metric under distribution shift, not as an i.i.d. coverage guarantee.

## Required outputs per run

Each run writes to `outputs/runs/<run_id>/`:
- `config.yaml`
- `meta.json`
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `calibration/temperature.json`, `calibration/conformal.json`
- `splits/split_manifest.json` -- stratified split indices with SHA-256 integrity hash
- `eval/<dataset>/metrics.json`
- `eval/<dataset>/predictions.parquet`
- `logs/train.jsonl`, `logs/eval.jsonl`

## Evaluation protocol

We evaluate:
- AUROC
- TPR @ 1% FPR
- ECE
- risk-coverage and AURC
- abstention rate
- selective/answered accuracy
- bootstrap confidence intervals (AUROC, selective accuracy)
- empirical conformal coverage (overall + class-conditional + per-group)
- worst-group selective accuracy (per-generator breakdown)

Baseline methods compared per run:

1. Forced decision (argmax)
2. Temperature-scaled forced decision
3. Confidence-threshold abstention (`max_softmax < tau`)
4. Split conformal (and optional Mondrian)
5. Threshold abstention (`max_softmax < τ`, via `--abstention-method threshold --tau 0.9`)

## Transform policy

`data.transforms` supports:
- `resize_shorter` and optional `center_crop`
- JPEG re-encoding quality perturbation
- Gaussian blur
- optional screenshot/re-post pipeline extension

Use deterministic mode (`deterministic: true`) to disable stochastic transforms for strict reproducibility.

## Notes on reproducibility and legal scope

- CFS licensing (CC BY-NC-SA) implies non-commercial/research constraints.
- Do not use outputs as legal evidence; the system is a reliability and screening tool.
- Report environment variables and exact command lines with every artifact bundle.

## Development

- [PLANS.md](PLANS.md) -- project roadmap and milestone tracking
- [AGENTS.md](AGENTS.md) -- agent skill descriptions and orchestration guidance
- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) -- detailed implementation status checklist

## Citation

If this code is used in reports or downstream work:

```bibtex
@misc{owaid_abstention_2026,
  title        = {Open-World AI Image Detection with Abstention},
  author       = {Ghabayen, Omar and Yar, Gorkem and Almansour, Abdullah and Xie, Zongchi and Qu, Youlin},
  year         = {2026},
  howpublished = {GitHub repository}
}
```
