"""Supervised training loop with log-spaced checkpoints, logged to the experiment DB."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from edge_ai_compression.benchmarking.evaluator import accuracy_on_loader
from edge_ai_compression.benchmarking.fingerprint import collect
from edge_ai_compression.core.experiment import dataset_num_classes
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.data.loaders import build_loaders
from edge_ai_compression.experiment_db.training_record import TrainingRecord, append_training_row
from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.utils.reproducibility import set_seed


def log_spaced_steps(total: int, n: int) -> list[int]:
    """``n`` roughly log-spaced step numbers in [1, total], always including ``total``."""
    steps = {int(round(s)) for s in np.geomspace(1, total, num=max(n, 1))}
    steps.add(total)
    return sorted(s for s in steps if 1 <= s <= total)


def lr_at(step: int, total: int, cfg: TrainConfig, steps_per_epoch: int) -> float:
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    if cfg.schedule == "constant":
        return cfg.lr
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    test_accuracy: float
    test_loss: float
    steps: int
    checkpoints: list[str]
    run_dir: Path


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    *,
    on_checkpoint: Any = None,
) -> tuple[int, list[dict[str, float]]]:
    """Train ``model`` in place. Returns (steps run, history rows).

    ``on_checkpoint(step, epoch)`` is called at every log-spaced checkpoint step.
    """
    device = torch.device(cfg.device)
    model.to(device)
    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    ckpt_steps = set(log_spaced_steps(total, cfg.num_checkpoints))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        exempt = p.ndim <= 1 or name.endswith((".bias", "_scale"))  # BN, biases, quant scales
        (no_decay if exempt else decay).append(p)
    opt = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov and cfg.momentum > 0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history: list[dict[str, float]] = []
    step = 0
    t0 = time.perf_counter()
    running = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for images, targets in train_loader:
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg, steps_per_epoch)
            images, targets = images.to(device), targets.to(device)
            loss = criterion(model(images), targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            running += loss.item()
            if step % cfg.log_every_steps == 0 or step == total:
                n = (
                    cfg.log_every_steps
                    if step % cfg.log_every_steps == 0
                    else step % cfg.log_every_steps
                )
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": running / n,
                        "lr": opt.param_groups[0]["lr"],
                        "seconds": time.perf_counter() - t0,
                    }
                )
                running = 0.0
            if step in ckpt_steps and on_checkpoint is not None:
                on_checkpoint(step, epoch)
    return step, history


def run_training(cfg: TrainConfig) -> TrainResult:
    set_seed(cfg.seed)
    fingerprint = collect()
    train_loader, test_loader = build_loaders(
        cfg.dataset,
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        limit_samples=cfg.limit_samples,
    )
    model = ModelRegistry.create(cfg.model, num_classes=dataset_num_classes(cfg.dataset))
    run_id = str(uuid.uuid4())
    run_dir = Path(cfg.results_dir) / "training" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.checkpoint_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[str] = []

    def save(step: int, epoch: int) -> None:
        path = ckpt_dir / f"step_{step:07d}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "step": step,
                "epoch": epoch,
                "run_id": run_id,
                "config": cfg.to_dict(),
            },
            path,
        )
        checkpoints.append(str(path))

    t0 = time.perf_counter()
    steps, history = train_model(model, train_loader, cfg, on_checkpoint=save)
    train_seconds = time.perf_counter() - t0
    acc, loss = accuracy_on_loader(model, test_loader, cfg.device)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    with open(run_dir / "history.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "fingerprint.json", "w", encoding="utf-8") as f:
        json.dump(fingerprint.to_dict(), f, indent=2, default=str)
    append_training_row(
        TrainingRecord.create(
            run_id=run_id,
            cfg=cfg,
            fingerprint=fingerprint,
            steps=steps,
            test_accuracy=acc,
            test_loss=loss,
            train_seconds=train_seconds,
            final_checkpoint=checkpoints[-1],
        ),
        Path(cfg.results_dir),
    )
    return TrainResult(run_id, acc, loss, steps, checkpoints, run_dir)
