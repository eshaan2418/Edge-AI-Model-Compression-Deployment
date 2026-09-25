from __future__ import annotations

import torch
import torch.nn as nn

from edge_ai_compression.compression.quantization import dynamic_quantize_linear_layers
from edge_ai_compression.utils.quant_engine import ensure_quantized_engine


def test_engine_selected():
    assert ensure_quantized_engine() != "none"


def test_dynamic_quantization_runs_and_is_close():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(4, 16)
    ref = m(x)
    q = dynamic_quantize_linear_layers(m)
    out = q(x)
    assert not isinstance(q[0], nn.Linear)
    assert torch.allclose(out, ref, atol=0.1)
