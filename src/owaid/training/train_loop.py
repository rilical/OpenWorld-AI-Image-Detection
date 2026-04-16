"""Minimal but explicit training loop and checkpoint management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm

from ..metrics.classification import auroc, tpr_at_fpr
from ..metrics.calibration_metrics import ece
from .checkpoints import load_checkpoint, save_checkpoint


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    amp: bool = False,
    grad_accum_steps: int = 1,
    logger=None,
    epoch: int = 0,
    global_step: int = 0,
    max_steps: int | None = None,
) -> Tuple[float, int]:
    """Train for one epoch and return average loss plus updated global step."""
    import itertools
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.startswith("cuda"))
    step_losses = []

    iterable = itertools.islice(loader, max_steps) if max_steps is not None else loader
    for batch_idx, batch in enumerate(tqdm(iterable, desc="train", leave=False, total=max_steps)):
        if batch is None:
            continue
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type="cuda", enabled=amp and device.startswith("cuda")):
            out = model(images)
            raw_loss = criterion(out["logits"], labels)
            loss = raw_loss / max(1, grad_accum_steps)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (batch_idx + 1) % max(1, grad_accum_steps) == 0 or batch_idx == len(loader) - 1
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        step_losses.append(float(raw_loss.item()))
        if logger is not None:
            logger.log(
                {
                    "split": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(raw_loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )

    avg_loss = float(sum(step_losses) / max(1, len(step_losses)))
    return avg_loss, global_step


def validate(model, loader, device: str, criterion, temperature: float | None = None) -> Dict[str, Any]:
    """Run validation and compute core metrics."""
    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            if batch is None:
                continue
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(images)
            all_logits.append(out["logits"])
            all_labels.append(labels)
            losses.append(float(criterion(out["logits"], labels).item()))

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    scaled_logits = logits / max(float(temperature or 1.0), 1e-8)
    probs = torch.softmax(scaled_logits, dim=-1)

    y_score = probs[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()
    ece_val, _ = ece(probs.cpu().numpy(), y_true)
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "auroc": auroc(y_true, y_score),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, y_score, fpr=0.01),
        "ece": ece_val,
    }


def run_training(
    cfg: Dict[str, Any],
    model,
    train_loader,
    val_loader,
    run_dir: str,
    device: str,
    logger,
    start_epoch: int = 0,
    best_metric: float = -1e9,
):
    """Run the configured training loop and persist checkpoints."""
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    amp = bool(train_cfg.get("amp", False))
    patience = train_cfg.get("early_stopping_patience")
    steps_per_epoch = train_cfg.get("steps_per_epoch")
    if steps_per_epoch is not None:
        steps_per_epoch = int(steps_per_epoch)
    metric_key = train_cfg.get("best_metric", "auroc")
    resume_from = train_cfg.get("resume_from")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    best_epoch = None
    no_improve_epochs = 0
    best_path = Path(run_dir) / "checkpoints" / "best.pt"
    last_path = Path(run_dir) / "checkpoints" / "last.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    
    # new for debugging
    metric_key = cfg.get("train", {}).get("best_metric", "auroc")

    if resume_from:
        state = load_checkpoint(str(resume_from), model, optimizer=optimizer)
        start_epoch = int(state.get("epoch", start_epoch)) + 1
        global_step = int(state.get("global_step", 0))

    if torch.cuda.is_available():
        print(f"[GPU] Using: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f"[GPU] Memory reserved:  {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
    else:
        print("[GPU] CUDA not available — running on CPU")

    for epoch in range(start_epoch, epochs):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp=amp,
            grad_accum_steps=grad_accum_steps,
            logger=logger,
            epoch=epoch,
            global_step=global_step,
            max_steps=steps_per_epoch,
        )
        logger.log(
            {
                "epoch": epoch,
                "split": "train",
                "loss": train_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        current_metric = train_loss
        val_metrics = None
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device, criterion)
            val_metrics.update({"epoch": epoch, "split": "val"})
            logger.log(val_metrics)
            current_metric = float(val_metrics.get(metric_key, -1e9))

        save_checkpoint(str(last_path), model, optimizer, epoch, global_step, cfg)

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(str(best_path), model, optimizer, epoch, global_step, cfg)
        else:
            no_improve_epochs += 1

        if patience is not None and no_improve_epochs >= int(patience):
            logger.log(
                {
                    "split": "train",
                    "event": "early_stop",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                }
            )
            break

    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": metric_key,
        "last_global_step": global_step,
    }
