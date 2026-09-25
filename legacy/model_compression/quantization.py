from __future__ import annotations

import torch
import torch.nn as nn


def dynamic_quantize_linear_layers(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to Linear layers for inference speed/size benefits."""
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
