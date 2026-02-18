# src/owaid/training/agents.md

Scope: training loop, evaluation loop, checkpointing.

Training loop:
- Minimal PyTorch loop (no heavy framework required)
- Supports:
  - mixed precision (optional)
  - gradient accumulation (optional)
  - early stopping (optional, config-controlled)
- Logs to JSONL:
  - step logs: loss, lr
  - epoch logs: val metrics (AUROC, ECE)

Checkpointing:
- Save:
  - best.pt (by val AUROC or val loss configurable)
  - last.pt
- Include:
  - model state_dict
  - optimizer state_dict (optional)
  - cfg snapshot
  - epoch, global_step

Evaluation loop:
- Produces:
  - logits/probs
  - metrics via src/owaid/metrics
  - predictions parquet (optional)
- Abstention evaluation is done here (not in training loop)
