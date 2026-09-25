"""Measured weight sparsity (no heavy imports: used by the runner and analyses)."""

from __future__ import annotations

import torch.nn as nn


def weight_sparsity(model: nn.Module) -> float:
    """Fraction of exactly-zero conv/linear weights (quantized wrappers included)."""
    from edge_ai_compression.compression.quantization.modules import QuantLayer

    total = zeros = 0
    for mod in model.modules():
        if not isinstance(mod, nn.Conv2d | nn.Linear | QuantLayer):
            continue
        w = mod.weight.detach()
        total += w.numel()
        zeros += int((w == 0).sum())
    return zeros / max(total, 1)
