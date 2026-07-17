from __future__ import annotations

import io
from typing import Any

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_size_mb(model: nn.Module) -> float:
    """Serialized ``state_dict`` size in MiB (a proxy for on-disk model size)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def sparsity(model: nn.Module) -> float:
    """Fraction of weight elements that are exactly zero (0.0 if no params)."""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p == 0).sum().item())
    return float(zeros) / total if total else 0.0


def model_summary(model: nn.Module) -> dict[str, Any]:
    """A JSON-serializable structural summary of a model.

    Captures parameter counts, a size estimate, weight sparsity, and a per-layer
    module-type histogram. Cheap and dependency-free (no forward pass).
    """
    layer_types: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        # Skip container modules that merely hold children.
        if name in ("Sequential", "ModuleList", "ModuleDict") or module is model:
            continue
        layer_types[name] = layer_types.get(name, 0) + 1

    num_params = count_parameters(model)
    return {
        "class_name": type(model).__name__,
        "num_parameters": num_params,
        "num_trainable_parameters": count_trainable_parameters(model),
        "size_mb": round(state_dict_size_mb(model), 6),
        "weight_sparsity": round(sparsity(model), 6),
        "num_modules": sum(layer_types.values()),
        "layer_type_histogram": dict(sorted(layer_types.items())),
    }


def estimate_flops_macs(model: nn.Module, input_shape: tuple[int, int, int, int]) -> float:
    """Rough MAC estimate for Conv2d/Linear on one forward (no BN/ReLU)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    class FlopHook:
        def __init__(self) -> None:
            self.total = 0.0

        def __call__(self, mod, inp, out) -> None:
            if isinstance(mod, nn.Conv2d):
                in_t = inp[0]
                b, cin, hin, win = in_t.shape
                k_h, k_w = mod.kernel_size
                out_t = out
                _, cout, hout, wout = out_t.shape
                self.total += b * hout * wout * cout * cin * k_h * k_w
            elif isinstance(mod, nn.Linear):
                in_t = inp[0]
                b, fin = in_t.shape
                fout = mod.out_features
                self.total += b * fin * fout

    hook = FlopHook()
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))

    dummy = torch.zeros(input_shape, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()
    return float(hook.total)


def weight_bytes(model: nn.Module, *, dtype_bytes: int = 4) -> float:
    """Total bytes occupied by the model's parameters at ``dtype_bytes`` each.

    Defaults to 4 bytes (FP32). Pass 1 for int8 to reason about a quantized model.
    """
    return float(count_parameters(model) * dtype_bytes)


def estimate_activation_bytes(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    *,
    dtype_bytes: int = 4,
) -> float:
    """Estimate bytes of intermediate activations produced in one forward pass.

    Sums the output-tensor sizes of every leaf module (the tensors a naive
    executor would materialize). This is an upper-ish estimate of activation
    memory traffic — real runtimes fuse and reuse buffers — but it is the right
    order of magnitude for roofline / arithmetic-intensity reasoning.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total = 0

    def hook(_mod, _inp, out) -> None:
        nonlocal total
        tensors = out if isinstance(out, (tuple, list)) else [out]
        for t in tensors:
            if isinstance(t, torch.Tensor):
                total += t.numel()

    handles = []
    for m in model.modules():
        # Leaf modules only (no children) — avoids double-counting containers.
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))

    dummy = torch.zeros(input_shape, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()
    return float(total * dtype_bytes)
