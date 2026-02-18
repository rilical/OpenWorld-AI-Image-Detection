# src/owaid/data/agents.md

Scope: dataset wrappers + transforms + dataloader builders.

Common sample schema (ALL datasets):
- __getitem__ returns dict:
  - "image": torch.FloatTensor [3,H,W] already normalized for CLIP
  - "label": int (0=Real, 1=AI) for binary datasets; for real-only datasets use label=0 and flag real_only
  - "meta": dict with stable keys when available:
      id, source_dataset, split, generator (optional), path (optional)

Transforms:
- build_clip_transform(cfg, train: bool) -> callable
- Optional corruption pipeline toggled by cfg:
  jpeg_quality, blur_sigma, resize_shorter, center_crop, etc.
- Must be deterministic when cfg.deterministic=True.

CommunityForensics-Small:
- load via Hugging Face datasets
- training uses official train split
- create calibration split by splitting train or val per cfg (document choice)
- store split indices in outputs so it’s reproducible

VCT2 / RAID / ARIA:
- provide robust stubs and clear env var requirements:
  VCT2_ROOT, ARIA_ROOT, etc.

Dataloader builders:
- build_commfor_dataloaders(cfg) -> dict[str, DataLoader]
- build_eval_dataloader(cfg, dataset_name) -> DataLoader
