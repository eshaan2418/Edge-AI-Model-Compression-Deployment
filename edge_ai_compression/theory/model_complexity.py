from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _gemm_layer(mod: nn.Module) -> str | None:
    from edge_ai_compression.compression.quantization.modules import QuantConv2d, QuantLinear

    if isinstance(mod, nn.Conv2d | QuantConv2d):
        return "conv"
    if isinstance(mod, nn.Linear | QuantLinear):
        return "linear"
    return None


def layer_gemm_shapes(model: nn.Module, input_shape: tuple[int, ...]) -> list[dict[str, object]]:
    """Per conv/linear layer (float or quantized), the GEMM it performs for one sample:
    M = output channels, K = inputs per output (in/groups * kh * kw), N = output
    positions (conv) or tokens (linear on [B, T, C] inputs, 1 for [B, C])."""
    shapes: list[dict[str, object]] = []

    def hook(name: str, kind: str):
        def fn(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            w = mod.weight
            m, k = int(w.shape[0]), int(w[0].numel())
            batch = inp[0].shape[0]
            n = out[0, 0].numel() if kind == "conv" else out.numel() // (batch * m)
            shapes.append({"name": name, "kind": kind, "m": m, "n": int(n), "k": k})

        return fn

    handles = [
        mod.register_forward_hook(hook(name, kind))
        for name, mod in model.named_modules()
        if (kind := _gemm_layer(mod)) is not None
    ]
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape, device=param.device, dtype=param.dtype))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return shapes


def estimate_flops_macs(model: nn.Module, input_shape: tuple[int, ...]) -> float:
    """Multiply-accumulates of all conv/linear layers (float or quantized) for one
    forward of ``input_shape`` (batch included). BN, activations and attention
    matmuls are not counted."""
    per_sample = sum(
        int(s["m"]) * int(s["n"]) * int(s["k"]) for s in layer_gemm_shapes(model, input_shape)
    )
    return float(input_shape[0] * per_sample)
