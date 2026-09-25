"""N:M semi-structured pruning (2:4 by default).

In every group of ``m`` consecutive weights along the flattened input dimension
([out, in * kh * kw], the layout the C++ engine uses), keep the ``n`` largest by
magnitude. Layers whose K is not divisible by ``m`` stay dense.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor | None:
    """Binary mask with exactly ``n`` ones per group of ``m``; None if K % m != 0."""
    if not 0 < n < m:
        raise ValueError(f"need 0 < n < m, got {n}:{m}")
    w2d = weight.detach().reshape(weight.shape[0], -1)
    k = w2d.shape[1]
    if k % m:
        return None
    groups = w2d.abs().reshape(w2d.shape[0], k // m, m)
    keep = groups.topk(n, dim=2).indices
    mask = torch.zeros_like(groups).scatter_(2, keep, 1.0)
    return mask.reshape(weight.shape)


def apply_nm_pruning(model: nn.Module, n: int = 2, m: int = 4) -> dict[str, str]:
    """Zero weights to N:M in place. Returns {layer: "pruned" | "skipped (K % m)"}."""
    report = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d | nn.Linear):
            continue
        mask = nm_mask(mod.weight, n, m)
        if mask is None:
            report[name] = f"skipped (K % {m} != 0)"
            continue
        with torch.no_grad():
            mod.weight.mul_(mask)
        report[name] = "pruned"
    return report


def satisfies_nm(weight: torch.Tensor, n: int, m: int) -> bool:
    """True if every group of ``m`` has at most ``n`` nonzeros."""
    w2d = weight.detach().reshape(weight.shape[0], -1)
    if w2d.shape[1] % m:
        return False
    nnz = (w2d.reshape(w2d.shape[0], -1, m) != 0).sum(dim=2)
    return bool((nnz <= n).all())
