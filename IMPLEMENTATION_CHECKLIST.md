# Implementation Status Checklist

## A) Agent map files

- [x] `agents.md` exists and contains the routing contract.
- [x] `src/owaid/utils/agents.md` exists and defines utility module contracts.
- [x] `src/owaid/data/agents.md` exists and defines dataset/transforms contracts.
- [x] `src/owaid/models/agents.md` exists and defines model contracts.
- [x] `src/owaid/training/agents.md` exists and defines loop/checkpoint/eval contracts.
- [x] `src/owaid/calibration/agents.md` exists and defines calibration contracts.
- [x] `src/owaid/metrics/agents.md` exists and defines metric contracts.
- [x] `scripts/agents.md` exists and defines CLI conventions.
- [x] `demo/agents.md` exists and defines demo contracts.
- [x] `configs/agents.md` exists and defines config schema.
- [x] `notebooks/agents.md` exists and defines notebook conventions.
- [x] `tests/agents.md` exists and defines testing contracts.

## B) Repo bootstrap + metadata

- [x] Directory tree created under `assets/`, `configs/`, `src/owaid/{data,models,training,calibration,metrics,utils}/`, `scripts/`, `notebooks/`, `demo/`, `outputs/{runs,plots}/`, `tests/`, `.github/workflows/`.
- [x] `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `CITATION.cff`, `LICENSE` exist.
- [x] `outputs/runs/.gitkeep` and `outputs/plots/.gitkeep` exist.
- [x] Editable install path works: `python3 -m pip install -e .` then `python3 -c "import owaid; print('ok')"` succeeds.
- [x] Compile check currently passes: `python3 -m compileall src`.

## C) Prompts 01–30

1. [x] Prompt 01/30 (Repo bootstrap + agent map): all required root/module files and packaging are present.
2. [x] Prompt 02/30 (Utilities): `config.py`, `paths.py`, `seed.py`, `logging.py` implemented with documented helpers.
3. [x] Prompt 03/30 (Run/meta helpers): `src/owaid/utils/run.py` exists with `make_run_dir`, `write_meta`, `save_resolved_config`.
4. [x] Prompt 04/30 (Transforms): `src/owaid/data/transforms.py` CLIP transform + JPEG/blur/crop path + self-test.
5. [x] Prompt 05/30 (CommunityForensics loader): deterministic split loader + calibration indices persisted.
6. [x] Prompt 06/30 (VCT2/RAID/ARIA stubs): dataset classes + loaders + env-var error paths implemented.
7. [x] Prompt 07/30 (Baseline model): `CLIPBinaryDetector` with OpenCLIP head, freeze/unfreeze, `encode`, `forward`.
8. [~] Prompt 08/30 (Training loop): core loop exists (`run_training`), but function shape/behavior is not exactly `train_one_run(cfg, model, loaders, run_dir)` contract.
9. [~] Prompt 09/30 (Evaluation loop): `evaluate_model` exists and writes predictions, but does not match exact required signature/branching behavior.
10. [~] Prompt 10/30 (Metrics AUROC/TPR): AUROC/TPR available, but `compute_basic_metrics` aggregator is not implemented.
11. [x] Prompt 11/30 (ECE): expected calibration error + reliability payload implemented.
12. [~] Prompt 12/30 (Risk-coverage/AURC): `risk_coverage`/`coverage`/`abstain_rate`/`selective_accuracy` exist; naming and expected `risk_coverage_curve`, `abstention_metrics` wrappers are missing.
13. [~] Prompt 13/30 (Temperature scaling): `fit_temperature` exists, but `apply_temperature` and read/write JSON helpers are missing.
14. [~] Prompt 14/30 (Split conformal): conformal quantile + set construction implemented, but naming (`compute_nonconformity`, `fit_conformal`, `coverage_eval`) is not fully aligned.
15. [~] Prompt 15/30 (Abstention policy): tri-state outputs exist via `predict_with_abstention`; explicit `tri_state_from_set`, `confidence_from_probs`, `apply_abstention` helpers absent.
16. [~] Prompt 16/30 (train_baseline script): script runs and trains, but does not yet emit a separate final `metrics.json` summary artifact as requested.
17. [~] Prompt 17/30 (calibrate_temperature script): implemented and runs; metadata and logging details are partially aligned with spec.
18. [~] Prompt 18/30 (build_conformal script): implemented; empirical coverage inclusion in conformal artifact is not explicitly persisted.
19. [~] Prompt 19/30 (eval_commfor): wrapper in place, but method-wise forced/temp/conformal side-by-side metrics are not explicitly split.
20. [~] Prompt 20/30 (eval_vct2): wrapper in place; robust/evaluation suite incomplete per prompt contract.
21. [~] Prompt 21/30 (eval_raid): wrapper in place; attack-success proxy and richer behavior not yet implemented.
22. [~] Prompt 22/30 (eval_aria): wrapper in place; real-only metric semantics and single-class-safe stats are not fully implemented.
23. [~] Prompt 23/30 (make_plots): generates simple bars; risk-coverage and reliability plots are not yet implemented.
24. [~] Prompt 24/30 (DIRE scaffolding): stubs exist and compile, but optional gating/explicit not-implemented messaging is partial.
25. [ ] Prompt 25/30 (cache_residuals script): file and deterministic cache pipeline not yet added.
26. [~] Prompt 26/30 (train_with_dire): script exists and uses fusion path, but explicit cache enforcement flow is incomplete.
27. [ ] Prompt 27/30 (Demo UI): `demo/app.py` is CLI inference, not Gradio tri-state UI.
28. [x] Prompt 28/30 (Configs): all requested YAML configs exist and are runnable.
29. [x] Prompt 29/30 (Notebooks): four notebook skeleton files exist.
30. [x] Prompt 30/30 (Tests + CI + README): `tests/`, `.github/workflows/ci.yml`, README polish exist; `python3 -m pytest -q` passes.
