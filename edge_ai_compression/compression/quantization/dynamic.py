from __future__ import annotations

import torch
import torch.nn as nn


def _ensure_qengine() -> None:
    """Select an available quantization backend if none is configured.

    On some builds (e.g. macOS/ARM wheels) the default engine is unset, which
    makes quantized ops fail with "NoQEngine". Pick the first supported engine
    (qnnpack on ARM, fbgemm on x86) so dynamic quantization works out of the box.
    """
    if torch.backends.quantized.engine != "none":
        return
    supported = list(torch.backends.quantized.supported_engines)
    for candidate in ("qnnpack", "fbgemm"):
        if candidate in supported:
            torch.backends.quantized.engine = candidate
            return


def dynamic_quantize_linear_layers(model: nn.Module) -> nn.Module:
    _ensure_qengine()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
