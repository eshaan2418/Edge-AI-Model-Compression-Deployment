from __future__ import annotations

import torch
import torch.nn as nn

from edge_ai_compression.utils.quant_engine import ensure_quantized_engine


def dynamic_quantize_linear_layers(model: nn.Module) -> nn.Module:
    ensure_quantized_engine()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
