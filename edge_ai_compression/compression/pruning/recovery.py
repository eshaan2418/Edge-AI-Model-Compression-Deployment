"""Accuracy recovery after pruning: masked full fine-tuning or masked LoRA.

The sparsity mask of every conv/linear layer is read from its zeros, so both
methods work after any pruning mode (unstructured, N:M, or channel-pruned dense).

- ``finetune``: all parameters train; masks are enforced by the
  ``torch.nn.utils.prune`` reparametrization (weight = weight_orig * mask).
- ``lora``: pruned weights are frozen; each conv/linear gets a low-rank update
  B @ A (rank r, B = 0 at init, scale alpha / r; Hu et al. 2021). The update is
  masked (W + M * scale * BA) so the merged layer keeps the sparsity pattern. An
  unmasked merge would silently densify it. Only adapters and biases train.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

METHODS = ("none", "finetune", "lora")


@dataclass(frozen=True)
class RecoveryConfig:
    method: str = "none"
    epochs: int = 1
    max_steps: int | None = None
    lr: float = 0.01
    weight_decay: float = 5e-4
    rank: int = 8
    lora_alpha: float = 16.0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"recovery.method must be one of {METHODS}")
        if self.rank < 1:
            raise ValueError("recovery.rank must be >= 1")

    @staticmethod
    def from_dict(d: dict) -> RecoveryConfig:
        unknown = set(d) - set(RecoveryConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown recovery keys: {sorted(unknown)}")
        return RecoveryConfig(**d)


def _prunable(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d | nn.Linear)]


def _train(model: nn.Module, loader: DataLoader, cfg: RecoveryConfig, device: str) -> None:
    # Imported lazily: pretraining imports core, whose runner imports this module.
    from edge_ai_compression.pretraining.config import TrainConfig
    from edge_ai_compression.pretraining.trainer import train_model

    train_cfg = TrainConfig(
        device=device,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_epochs=0.0,
        schedule="cosine",
        log_every_steps=10,
    )
    train_model(model, loader, train_cfg)


def finetune_masked(model: nn.Module, loader: DataLoader, cfg: RecoveryConfig, device: str) -> None:
    layers = _prunable(model)
    for _, mod in layers:
        prune.custom_from_mask(mod, "weight", (mod.weight.detach() != 0).to(mod.weight.dtype))
    try:
        _train(model, loader, cfg, device)
    finally:
        for _, mod in layers:
            with contextlib.suppress(ValueError):
                prune.remove(mod, "weight")
    model.eval()


class LoRALayer(nn.Module):
    """Frozen (sparse) conv/linear plus a masked low-rank update."""

    def __init__(self, base: nn.Conv2d | nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        w2d = base.weight.detach().reshape(base.weight.shape[0], -1)
        self.register_buffer("mask", (w2d != 0).to(w2d.dtype))
        self.lora_a = nn.Parameter(torch.empty(rank, w2d.shape[1]))
        self.lora_b = nn.Parameter(torch.zeros(w2d.shape[0], rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        base.weight.requires_grad_(False)

    def delta(self) -> torch.Tensor:
        return (self.mask * (self.lora_b @ self.lora_a) * self.scale).reshape(
            self.base.weight.shape
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base.weight + self.delta()
        if isinstance(self.base, nn.Conv2d):
            b = self.base
            return F.conv2d(x, w, b.bias, b.stride, b.padding, b.dilation, b.groups)
        return F.linear(x, w, self.base.bias)

    def merged(self) -> nn.Conv2d | nn.Linear:
        with torch.no_grad():
            self.base.weight.add_(self.delta())
        self.base.weight.requires_grad_(True)
        return self.base


def _set(model: nn.Module, path: str, new: nn.Module) -> None:
    parent, _, attr = path.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, attr, new)


def lora_recover(model: nn.Module, loader: DataLoader, cfg: RecoveryConfig, device: str) -> int:
    """Train masked LoRA adapters, then merge them. Returns the adapter parameter count."""
    layers = _prunable(model)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in layers:
        lora = LoRALayer(mod, cfg.rank, cfg.lora_alpha).to(mod.weight.device)
        if mod.bias is not None:
            mod.bias.requires_grad_(True)
        _set(model, name, lora)
        wrapped.append((name, lora))
    n_adapter = sum(lora.lora_a.numel() + lora.lora_b.numel() for _, lora in wrapped)
    _train(model, loader, cfg, device)
    for name, lora in wrapped:
        _set(model, name, lora.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return n_adapter


def recover(model: nn.Module, loader: DataLoader | None, cfg: RecoveryConfig, device: str) -> None:
    if cfg.method == "none":
        return
    if loader is None:
        raise ValueError("recovery needs training data (a train loader)")
    if cfg.method == "finetune":
        finetune_masked(model, loader, cfg, device)
    else:
        lora_recover(model, loader, cfg, device)
