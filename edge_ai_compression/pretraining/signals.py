"""Compressibility signals measured during training (Phase 5 surrogate features).

All signals use a fixed probe batch drawn once from the *training* data, so no
test-set information leaks into the Phase 6 early-prediction study.

Per conv/linear layer:
- weight kurtosis E[(w-mu)^4] / sigma^4 (uniform 1.8, Gaussian 3; heavy tails hurt
  quantization; Shkolnik et al. 2020)
- weight outlier ratio max|w| / std(w) (sets the min-max quantization step)
- weight L2 norm
- input-activation outlier ratio max|x| / std(x), and the per-channel max / median
  ratio used to motivate SmoothQuant (Xiao et al. 2023)
Model-level:
- Hutchinson trace of the loss Hessian (sum of per-layer block traces)
- sharpness L(w + rho * g / ||g||) - L(w), the first-order worst case of SAM's
  inner maximization (Foret et al. 2021)
- probe loss and weight sparsity
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_ai_compression.analysis.hessian import layer_traces


@dataclass(frozen=True)
class SignalConfig:
    probe_samples: int = 256
    hessian: bool = True
    hessian_iters: int = 20
    sharpness: bool = True
    rho: float = 0.05

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SignalConfig:
        unknown = set(d) - set(SignalConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown signal options: {sorted(unknown)}")
        return SignalConfig(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCALAR_FIELDS = (
    "probe_loss",
    "kurtosis_mean",
    "kurtosis_max",
    "w_outlier_mean",
    "w_outlier_max",
    "weight_l2_mean",
    "act_outlier_mean",
    "act_outlier_max",
    "act_channel_ratio_mean",
    "act_channel_ratio_max",
    "hessian_trace",
    "sharpness",
    "weight_sparsity",
)


def _layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d | nn.Linear)]


def kurtosis(w: torch.Tensor) -> float:
    w = w.detach().float().reshape(-1)
    c = w - w.mean()
    var = c.pow(2).mean()
    return float(c.pow(4).mean() / var.pow(2).clamp_min(1e-24))


def weight_signals(model: nn.Module) -> dict[str, dict[str, float]]:
    out = {}
    for name, mod in _layers(model):
        w = mod.weight.detach().float()
        std = float(w.std()) or 1e-12
        out[name] = {
            "kurtosis": kurtosis(w),
            "w_outlier": float(w.abs().max()) / std,
            "weight_l2": float(w.norm()),
        }
    return out


@torch.no_grad()
def activation_signals(model: nn.Module, x: torch.Tensor) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}

    def hook(name: str, is_conv: bool):
        def fn(_m: nn.Module, args: tuple) -> None:
            a = args[0].detach().float()
            ch = (
                a.transpose(0, 1).reshape(a.shape[1], -1)
                if is_conv
                else a.reshape(-1, a.shape[-1]).T
            )
            per_channel = ch.abs().amax(dim=1)
            stats[name] = {
                "act_outlier": float(a.abs().max()) / (float(a.std()) or 1e-12),
                "act_channel_ratio": float(
                    per_channel.max() / per_channel.median().clamp_min(1e-12)
                ),
            }

        return fn

    hooks = [
        mod.register_forward_pre_hook(hook(name, isinstance(mod, nn.Conv2d)))
        for name, mod in _layers(model)
    ]
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return stats


def sharpness(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, rho: float
) -> tuple[float, float]:
    """(loss, first-order worst-case loss increase within an L2 ball of radius rho)."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    norm = torch.sqrt(sum(g.pow(2).sum() for g in grads)).clamp_min(1e-12)
    with torch.no_grad():
        saved = [p.detach().clone() for p in params]  # (w + e) - e is not exact in float
        for p, g in zip(params, grads, strict=True):
            p.add_(rho * g / norm)
        perturbed = F.cross_entropy(model(x), y)
        for p, w in zip(params, saved, strict=True):
            p.copy_(w)
    return float(loss), float(perturbed - loss)


def compute_signals(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: SignalConfig
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Scalar summaries (SCALAR_FIELDS) and per-layer detail. Restores the model's mode."""
    was_training = model.training
    model.eval()
    try:
        per_layer = weight_signals(model)
        for name, vals in activation_signals(model, x).items():
            per_layer[name].update(vals)
        if cfg.sharpness:
            loss, sharp = sharpness(model, x, y, cfg.rho)
        else:
            sharp = float("nan")
            with torch.no_grad():
                loss = float(F.cross_entropy(model(x), y))
        trace = float("nan")
        if cfg.hessian:
            traces = layer_traces(model, x, y, list(per_layer), max_iters=cfg.hessian_iters)
            trace = float(sum(traces.values()))
            for name, t in traces.items():
                per_layer[name]["hessian_trace"] = t
    finally:
        model.train(was_training)

    def agg(key: str, reduce: Callable[[torch.Tensor], torch.Tensor]) -> float:
        return float(reduce(torch.tensor([v[key] for v in per_layer.values()])))

    total = sum(m.weight.numel() for _, m in _layers(model))
    zeros = sum(int((m.weight == 0).sum()) for _, m in _layers(model))
    scalars = {
        "probe_loss": loss,
        "kurtosis_mean": agg("kurtosis", torch.mean),
        "kurtosis_max": agg("kurtosis", torch.max),
        "w_outlier_mean": agg("w_outlier", torch.mean),
        "w_outlier_max": agg("w_outlier", torch.max),
        "weight_l2_mean": agg("weight_l2", torch.mean),
        "act_outlier_mean": agg("act_outlier", torch.mean),
        "act_outlier_max": agg("act_outlier", torch.max),
        "act_channel_ratio_mean": agg("act_channel_ratio", torch.mean),
        "act_channel_ratio_max": agg("act_channel_ratio", torch.max),
        "hessian_trace": trace,
        "sharpness": sharp,
        "weight_sparsity": zeros / max(total, 1),
    }
    return scalars, per_layer
