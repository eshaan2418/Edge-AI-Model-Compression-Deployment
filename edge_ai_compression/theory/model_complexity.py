from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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
