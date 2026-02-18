# configs/agents.md

Scope: YAML experiment config schema.

Recommended top-level keys:
- seed: int
- deterministic: bool
- device: "cuda" | "cpu"
- output:
    root: "outputs/runs"
    run_name: str (optional)
- data:
    dataset: "commfor_small" | "vct2" | "raid" | "aria"
    batch_size: int
    num_workers: int
    img_size: int
    calibration_fraction: float
    transforms:
      use_corruptions: bool
      jpeg_quality: int|None
      blur_sigma: float|None
      resize_shorter: int|None
- model:
    type: "clip_baseline" | "clip_dire_fusion"
    clip:
      model_name: str
      pretrained: str
      freeze: bool
      unfreeze_last_n: int|None
    head:
      hidden_dims: [int, ...]
      dropout: float
    dire:
      enabled: bool
      cache_dir: str|None
- train:
    epochs: int
    lr: float
    weight_decay: float
    amp: bool
    grad_accum_steps: int
    best_metric: "auroc" | "loss"
- calibration:
    temperature: bool
    conformal_alpha: float
    conformal_method: "split" | "mondrian"
- eval:
    save_predictions: bool
