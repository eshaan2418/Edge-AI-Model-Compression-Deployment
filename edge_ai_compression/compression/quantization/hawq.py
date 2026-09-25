"""HAWQ-style mixed-precision bit allocation (Dong et al., HAWQ-V2, NeurIPS 2020).

Per-layer sensitivity to b-bit quantization is Omega_i(b) = (tr(H_i) / n_i) *
||Q_b(W_i) - W_i||^2, with tr(H_i) the Hutchinson trace of the layer's Hessian
block (average curvature times quantization perturbation). Bits are chosen by an
exact integer linear program (as in HAWQ-V3) minimizing total sensitivity
subject to an average-bit budget over the weights. Pinned layers (first/last)
keep their bits but count toward the budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from edge_ai_compression.compression.quantization.quantizer import (
    WeightSpec,
    fake_quant_weight,
    weight_scale,
)


@dataclass(frozen=True)
class HAWQConfig:
    candidate_bits: tuple[int, ...] = (4, 8)
    avg_bits: float = 6.0
    hutchinson_iters: int = 100
    hutchinson_samples: int = 256
    tol: float = 1e-3

    @staticmethod
    def from_dict(d: dict) -> HAWQConfig:
        allowed = {"candidate_bits", "avg_bits", "hutchinson_iters", "hutchinson_samples", "tol"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown hawq options: {sorted(unknown)}")
        kw = dict(d)
        if "candidate_bits" in kw:
            kw["candidate_bits"] = tuple(int(b) for b in kw["candidate_bits"])
        return HAWQConfig(**kw)


def sensitivity(w: torch.Tensor, trace: float, bits: int, spec: WeightSpec) -> float:
    s = WeightSpec(
        bits=bits, granularity=spec.granularity, group_size=spec.group_size, method=spec.method
    )
    err = float((fake_quant_weight(w, weight_scale(w, s), s) - w).pow(2).sum())
    return trace / w.numel() * err


def allocate_bits(
    weights: dict[str, torch.Tensor],
    traces: dict[str, float],
    cfg: HAWQConfig,
    spec: WeightSpec,
    pinned: dict[str, int] | None = None,
) -> dict[str, int]:
    """Solve min sum Omega_i(b_i) s.t. sum n_i b_i <= avg_bits * sum n_i (exact ILP)."""
    pinned = pinned or {}
    free = [n for n in weights if n not in pinned]
    bits = list(cfg.candidate_bits)
    sizes = {n: weights[n].numel() for n in weights}
    budget = cfg.avg_bits * sum(sizes.values()) - sum(sizes[n] * b for n, b in pinned.items())
    if not free:
        return dict(pinned)
    n_free, n_bits = len(free), len(bits)
    cost = np.array(
        [sensitivity(weights[n].detach(), traces[n], b, spec) for n in free for b in bits]
    )
    one_hot = np.zeros((n_free, n_free * n_bits))
    for i in range(n_free):
        one_hot[i, i * n_bits : (i + 1) * n_bits] = 1
    size_row = np.array([[sizes[n] * b for n in free for b in bits]])
    res = milp(
        c=cost,
        constraints=[LinearConstraint(one_hot, 1, 1), LinearConstraint(size_row, -np.inf, budget)],
        integrality=np.ones_like(cost),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise ValueError(f"no bit allocation fits avg_bits={cfg.avg_bits}: {res.message}")
    choice = res.x.reshape(n_free, n_bits).argmax(axis=1)
    return {**pinned, **{n: bits[int(c)] for n, c in zip(free, choice, strict=True)}}
