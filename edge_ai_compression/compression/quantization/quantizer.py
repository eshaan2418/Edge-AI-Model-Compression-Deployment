"""Core quantization math shared by every rung of the PTQ/QAT ladder.

All grids are symmetric and narrow-range: b bits -> integers in
[-(2^(b-1) - 1), 2^(b-1) - 1] (8-bit: [-127, 127], 4-bit: [-7, 7]), matching the
C++ kernels (DECISIONS D2.3). Scales are per tensor, per output channel (axis
0), or per group of ``group_size`` consecutive input elements within an output
channel (weights flattened to [out, in*kh*kw], as the engine lays them out).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GRANULARITIES = ("per_tensor", "per_channel", "per_group")


def qmax(bits: int) -> int:
    if not 2 <= bits <= 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")
    return 2 ** (bits - 1) - 1


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """round() with a straight-through gradient."""
    return x + (torch.round(x) - x).detach()


def fake_quant(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric fake quantization with STE through rounding (and LSQ-style
    gradients to ``scale`` when it requires grad). ``scale`` must broadcast to x."""
    q = qmax(bits)
    # Clamp before rounding (LSQ): gradient flows iff x / scale is inside the grid range.
    return round_ste(torch.clamp(x / scale, -q, q)) * scale


def quantize_int(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    q = qmax(bits)
    return torch.clamp(torch.round(x / scale), -q, q).to(torch.int32)


@dataclass(frozen=True)
class WeightSpec:
    bits: int = 8
    granularity: str = "per_channel"
    group_size: int | None = None  # per_group only
    method: str = "minmax"  # minmax | mse

    def __post_init__(self) -> None:
        qmax(self.bits)
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}")
        if (self.granularity == "per_group") != (self.group_size is not None):
            raise ValueError("group_size is required for (and only for) per_group")
        if self.method not in ("minmax", "mse"):
            raise ValueError("weight method must be minmax or mse")


def _grouped(w2d: torch.Tensor, group: int) -> torch.Tensor:
    """[M, K] -> [M, ceil(K/group), group] (zero-padded)."""
    m, k = w2d.shape
    n = -(-k // group)
    padded = torch.zeros(m, n * group, dtype=w2d.dtype, device=w2d.device)
    padded[:, :k] = w2d
    return padded.reshape(m, n, group)


def weight_scale(w: torch.Tensor, spec: WeightSpec) -> torch.Tensor:
    """Scale tensor broadcastable to the [M, K] flattened weight (or [M, G, group])."""
    w2d = w.detach().reshape(w.shape[0], -1)
    if spec.granularity == "per_tensor":
        amax = w2d.abs().max().reshape(1, 1)
        data = w2d
    elif spec.granularity == "per_channel":
        amax = w2d.abs().amax(dim=1, keepdim=True)
        data = w2d
    else:
        assert spec.group_size is not None
        data = _grouped(w2d, spec.group_size)
        amax = data.abs().amax(dim=2, keepdim=True)
    amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale = amax / qmax(spec.bits)
    if spec.method == "mse":
        scale = _mse_scale(data, scale, spec.bits)
    return scale


def _mse_scale(data: torch.Tensor, base: torch.Tensor, bits: int, steps: int = 80) -> torch.Tensor:
    """Per-slice clipping ratio in [0.2, 1] minimizing ||x - Q(x)||^2 (grid search)."""
    best = base.clone()
    best_err = torch.full_like(base, float("inf"))
    for ratio in torch.linspace(0.2, 1.0, steps):
        s = base * ratio
        err = (fake_quant(data, s, bits) - data).pow(2)
        # One scale per slice along the last dim (channel or group), or one overall.
        err = err.sum(dim=-1, keepdim=True) if base.numel() > 1 else err.sum().reshape_as(base)
        better = err < best_err
        best = torch.where(better, s, best)
        best_err = torch.where(better, err, best_err)
    return best


def scale_2d(scale: torch.Tensor, spec: WeightSpec, m: int, k: int) -> torch.Tensor:
    """Expand a compact scale from ``weight_scale`` to one scale per element of [M, K]."""
    if spec.granularity == "per_group":
        assert spec.group_size is not None
        return scale.reshape(m, -1).repeat_interleave(spec.group_size, dim=1)[:, :k]
    return (
        scale.reshape(-1, 1).expand(m, k) if scale.numel() > 1 else scale.reshape(1, 1).expand(m, k)
    )


def fake_quant_weight(w: torch.Tensor, scale: torch.Tensor, spec: WeightSpec) -> torch.Tensor:
    """Fake-quantize a weight of any shape with a scale from ``weight_scale``."""
    w2d = w.reshape(w.shape[0], -1)
    if spec.granularity == "per_group":
        assert spec.group_size is not None
        k = w2d.shape[1]
        q = fake_quant(_grouped(w2d, spec.group_size), scale, spec.bits)
        return q.reshape(w2d.shape[0], -1)[:, :k].reshape(w.shape)
    return fake_quant(w2d, scale, spec.bits).reshape(w.shape)


def integer_weight(w: torch.Tensor, scale: torch.Tensor, spec: WeightSpec) -> torch.Tensor:
    """Integer codes [M, K] matching ``fake_quant_weight``."""
    w2d = w.detach().reshape(w.shape[0], -1)
    if spec.granularity == "per_group":
        assert spec.group_size is not None
        k = w2d.shape[1]
        q = quantize_int(_grouped(w2d, spec.group_size), scale, spec.bits)
        return q.reshape(w2d.shape[0], -1)[:, :k]
    return quantize_int(w2d, scale, spec.bits)


# ------------------------------------------------------------ observers ----


class ActObserver:
    """Collects activation statistics during calibration and produces a
    symmetric per-tensor scale.

    minmax: max |x|.  percentile: the p-th percentile of |x|.  mse: the clipping
    value in [0.2, 1] x max|x| minimizing quantization MSE. percentile and mse
    keep a uniform random sample of at most ``max_samples`` values.
    """

    def __init__(
        self,
        method: str = "minmax",
        bits: int = 8,
        percentile: float = 99.99,
        max_samples: int = 1_000_000,
        seed: int = 0,
    ) -> None:
        if method not in ("minmax", "percentile", "mse"):
            raise ValueError("observer method must be minmax, percentile, or mse")
        self.method, self.bits, self.percentile = method, bits, percentile
        self.max_samples = max_samples
        self.amax = 0.0
        self.samples: list[torch.Tensor] = []
        self.n_seen = 0
        self.gen = torch.Generator().manual_seed(seed)

    def observe(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1).cpu()
        self.amax = max(self.amax, float(x.abs().max()) if x.numel() else 0.0)
        if self.method == "minmax":
            return
        self.n_seen += x.numel()
        keep = min(x.numel(), max(1, self.max_samples // 8))
        idx = torch.randint(0, x.numel(), (keep,), generator=self.gen)
        self.samples.append(x[idx])
        total = sum(s.numel() for s in self.samples)
        if total > self.max_samples:
            flat = torch.cat(self.samples)
            idx = torch.randperm(flat.numel(), generator=self.gen)[: self.max_samples]
            self.samples = [flat[idx]]

    def scale(self) -> float:
        if self.amax == 0.0:
            return 1.0
        q = qmax(self.bits)
        if self.method == "minmax":
            return self.amax / q
        data = torch.cat(self.samples)
        if self.method == "percentile":
            clip = float(torch.quantile(data.abs(), self.percentile / 100.0))
            return max(clip, 1e-12) / q
        base = torch.tensor([[self.amax / q]])
        return float(_mse_scale(data.reshape(1, -1), base, self.bits))
