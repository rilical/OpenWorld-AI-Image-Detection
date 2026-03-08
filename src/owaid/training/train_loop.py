"""Minimal training loop and checkpointing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm

from ..metrics.classification import auroc, tpr_at_fpr
from ..metrics.calibration_metrics import ece
from ..metrics.selective import risk_coverage
from ..models.abstention import predict_with_abstention

from .checkpoints import save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, amp: bool = False, grad_accum_steps: int = 1):
    model.train()
    step_losses = []
    total_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        out = model(images)
        loss = criterion(out["logits"], labels)
        loss = loss / max(1, grad_accum_steps)

        loss.backward()
        total_steps += 1
        if total_steps % max(1, grad_accum_steps) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step_losses.append(float(loss.item()) * max(1, grad_accum_steps))

    return float(sum(step_losses) / max(1, len(step_losses)))


def validate(model, loader, device, criterion, temperature: float | None = None, conformal: Dict[str, Any] | None = None):
    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(images)
            all_logits.append(out["logits"]) 
            all_labels.append(labels)
            losses.append(float(criterion(out["logits"], labels).item()))

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1) if temperature else torch.softmax(logits, dim=-1)

    y_score = probs[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()

    metrics = {
        "loss": float(sum(losses) / max(1, len(losses))),
        "auroc": auroc(y_true, y_score),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, y_score, fpr=0.01),
        "coverage": 1.0,
    }

    ece_val, ece_payload = ece(probs.cpu().numpy(), y_true)
    metrics.update({"ece": ece_val})
    return metrics


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
    train_cfg = cfg.get("train", cfg)
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    amp = bool(train_cfg.get("amp", False))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    best_path = Path(run_dir) / "checkpoints" / "best.pt"
    last_path = Path(run_dir) / "checkpoints" / "last.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    
    # new for debugging
    metric_key = cfg.get("train", {}).get("best_metric", "auroc")

    for epoch in range(start_epoch, epochs):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, amp=amp, grad_accum_steps=grad_accum_steps)
        logger.log(
            {
                "epoch": epoch,
                "split": "train",
                "loss": tr_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loader is None:
            save_checkpoint(str(last_path), model, optimizer, epoch, global_step, cfg) # new for debugging
            global_step += 1 # new for debugging
            continue

        val_metrics = validate(model, val_loader, device, criterion)
        val_metrics.update({"epoch": epoch, "split": "val"})
        logger.log(val_metrics)

        metric_key = cfg.get("train", {}).get("best_metric", "auroc")
        metric_value = val_metrics.get(metric_key, -1)
        if metric_value is not None and metric_value > best_metric:
            best_metric = metric_value
            save_checkpoint(str(best_path), model, optimizer, epoch, global_step, cfg)

        save_checkpoint(str(last_path), model, optimizer, epoch, global_step, cfg)
        global_step += 1

    return {
        "epochs": epochs,
        "best_metric": best_metric,
        "best_metric_name": metric_key,
    }
