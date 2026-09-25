"""Static activation calibration and calibration-data helpers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.compression.quantization.modules import quant_layers
from edge_ai_compression.compression.quantization.quantizer import ActObserver


def calibration_batches(
    loader: DataLoader, num_samples: int, device: str = "cpu"
) -> Iterator[torch.Tensor]:
    """Yield input batches from ``loader`` until ``num_samples`` images were produced."""
    seen = 0
    for images, _ in loader:
        if seen >= num_samples:
            return
        take = min(len(images), num_samples - seen)
        seen += take
        yield images[:take].to(device)


def calibration_tensor(loader: DataLoader, num_samples: int, device: str = "cpu") -> torch.Tensor:
    return torch.cat(list(calibration_batches(loader, num_samples, device)))


@torch.no_grad()
def calibrate_activations(
    model: nn.Module,
    loader: DataLoader,
    *,
    method: str = "minmax",
    num_samples: int = 512,
    percentile: float = 99.99,
    device: str = "cpu",
) -> dict[str, float]:
    """Set static input scales on every QuantLayer that quantizes activations.

    Statistics are collected with activation quantization disabled everywhere
    (inputs seen in float, weights already fake-quantized), then each layer's
    scale is set from its observer and activation quantization is enabled.
    Returns {layer name: scale}.
    """
    layers = [(n, m) for n, m in quant_layers(model) if m.act_bits is not None]
    for _, layer in layers:
        layer.act_enabled = False
    observers = {
        n: ActObserver(method, bits=m.act_bits or 8, percentile=percentile) for n, m in layers
    }
    hooks = [
        m.register_forward_pre_hook(lambda _mod, args, name=n: observers[name].observe(args[0]))
        for n, m in layers
    ]
    model.eval().to(device)
    try:
        for x in calibration_batches(loader, num_samples, device):
            model(x)
    finally:
        for h in hooks:
            h.remove()
    scales = {}
    for name, layer in layers:
        scales[name] = observers[name].scale()
        layer.a_scale.fill_(scales[name])
        layer.act_enabled = True
    return scales
