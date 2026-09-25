"""Quantization-aware fine-tuning with LSQ learnable step sizes (Esser et al., ICLR 2020).

Starts from a calibrated PTQ model: weight and activation scales become
learnable parameters (with LSQ's 1/sqrt(numel * qmax) gradient scaling) and the
network is fine-tuned through straight-through rounding. BatchNorm stays folded
into the convs during fine-tuning (DECISIONS D3.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.compression.quantization.modules import quant_layers
from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.trainer import train_model


@dataclass(frozen=True)
class QATConfig:
    epochs: int = 1
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-5
    max_steps: int | None = None

    @staticmethod
    def from_dict(d: dict) -> QATConfig:
        unknown = set(d) - set(QATConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown qat options: {sorted(unknown)}")
        return QATConfig(**d)


def quantization_aware_finetune(
    model: nn.Module, loader: DataLoader, cfg: QATConfig, device: str
) -> list[dict[str, float]]:
    """Enable LSQ on every QuantLayer and fine-tune in place. Returns the loss history."""
    for _, layer in quant_layers(model):
        layer.enable_lsq()
    train_cfg = TrainConfig(
        device=device,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=0.0,
        schedule="cosine",
        log_every_steps=10,
    )
    _, history = train_model(model, loader, train_cfg)
    model.eval()
    return history
