from __future__ import annotations

import torch
import torch.nn as nn


def dynamic_quantize_linear_layers(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
