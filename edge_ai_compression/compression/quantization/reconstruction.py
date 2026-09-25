"""AdaRound and BRECQ: learned rounding by output reconstruction.

Both learn, for every weight, whether to round down or up (AdaRound's rectified
sigmoid h(alpha), annealed to hard 0/1 by a regularizer) so that a *unit*'s
output on calibration data matches the float model:

- AdaRound (Nagel et al., ICML 2020): a unit is one conv / linear layer.
- BRECQ (Li et al., ICLR 2021): a unit is one residual block (plus the stem and
  classifier on their own), so rounding errors inside a block can cancel. The
  loss can be weighted by the diagonal Fisher (squared gradient of the task
  loss w.r.t. the unit output), BRECQ's approximation of the output Hessian.

Target = float model's unit output on float inputs. Input = what the partially
quantized model actually feeds the unit, so earlier units' errors are
compensated ("asymmetric" reconstruction). Weights and biases stay fixed; only
rounding logits are optimized. Activation quantization (static scales,
calibrated beforehand) is active during reconstruction, as at deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edge_ai_compression.compression.quantization.modules import QuantLayer, quant_layers

BLOCK_TYPES = ("BasicBlock", "Bottleneck")


@dataclass(frozen=True)
class ReconstructionConfig:
    iters: int = 2000
    lr: float = 1e-3
    reg_weight: float = 0.01
    batch_size: int = 32
    num_samples: int = 512
    beta_start: float = 20.0
    beta_end: float = 2.0
    warmup: float = 0.2
    fisher: bool = False  # BRECQ only: weight the loss by the diagonal Fisher

    @staticmethod
    def from_dict(d: dict) -> ReconstructionConfig:
        known = set(ReconstructionConfig.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown reconstruction options: {sorted(unknown)}")
        return ReconstructionConfig(**d)


def units(model: nn.Module, granularity: str) -> list[str]:
    """Module names to reconstruct, in forward order.

    ``layer``: every QuantLayer. ``block``: every residual block, plus
    QuantLayers not inside one (e.g. the stem conv and the classifier).
    """
    if granularity not in ("layer", "block"):
        raise ValueError("granularity must be 'layer' or 'block'")
    layers = [n for n, _ in quant_layers(model)]
    if granularity == "layer":
        return layers
    blocks = [n for n, m in model.named_modules() if type(m).__name__ in BLOCK_TYPES]
    order: list[str] = []
    for name in layers:
        block = next((b for b in blocks if name.startswith(b + ".")), None)
        unit = block or name
        if unit not in order:
            order.append(unit)
    return order


def _capture(
    model: nn.Module, unit: nn.Module, data: torch.Tensor, batch: int, *, want_input: bool
) -> torch.Tensor:
    """Input (or output) of ``unit`` for every row of ``data``."""
    seen: list[torch.Tensor] = []

    def hook(_m: nn.Module, args: tuple, out: torch.Tensor) -> None:
        seen.append((args[0] if want_input else out).detach())

    h = unit.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, len(data), batch):
                model(data[i : i + batch])
    finally:
        h.remove()
    return torch.cat(seen)


def _fisher(
    model: nn.Module, unit: nn.Module, data: torch.Tensor, labels: torch.Tensor, batch: int
) -> torch.Tensor:
    """Squared gradient of cross-entropy w.r.t. the unit output (diagonal Fisher)."""
    grads: list[torch.Tensor] = []
    outs: list[torch.Tensor] = []
    h = unit.register_forward_hook(lambda _m, _a, out: outs.append(out) or out.retain_grad())
    try:
        for i in range(0, len(data), batch):
            outs.clear()
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(data[i : i + batch]), labels[i : i + batch]).backward()
            grads.append(outs[0].grad.detach().pow(2))
    finally:
        h.remove()
    return torch.cat(grads)


def _reconstruct_unit(
    unit_q: nn.Module,
    x_q: torch.Tensor,
    y_fp: torch.Tensor,
    weight: torch.Tensor | None,
    cfg: ReconstructionConfig,
    seed: int,
) -> dict[str, float]:
    layers = [m for m in unit_q.modules() if isinstance(m, QuantLayer)]
    frozen = [p for p in unit_q.parameters()]
    for p in frozen:
        p.requires_grad_(False)
    for layer in layers:
        layer.init_adaround()
    opt = torch.optim.Adam([layer.alpha for layer in layers], lr=cfg.lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    warm = int(cfg.warmup * cfg.iters)
    stats = {"rec_loss_first": float("nan"), "rec_loss_last": float("nan")}
    unit_q.train(False)
    for it in range(cfg.iters):
        idx = torch.randint(0, len(x_q), (min(cfg.batch_size, len(x_q)),), generator=gen)
        err = (unit_q(x_q[idx]) - y_fp[idx]).pow(2)
        if weight is not None:
            err = err * weight[idx]
        rec = err.flatten(1).sum(1).mean()
        loss = rec
        if it >= warm:
            t = (it - warm) / max(cfg.iters - warm, 1)
            beta = cfg.beta_end + 0.5 * (cfg.beta_start - cfg.beta_end) * (
                1 + math.cos(math.pi * t)
            )
            reg = sum((1 - (2 * layer.rounding() - 1).abs().pow(beta)).sum() for layer in layers)
            loss = rec + cfg.reg_weight * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it == 0:
            stats["rec_loss_first"] = float(rec)
        stats["rec_loss_last"] = float(rec)
    for layer in layers:
        layer.hard_round = True
        assert layer.alpha is not None
        layer.alpha.requires_grad_(False)
    for p in frozen:
        p.requires_grad_(True)
    return stats


def reconstruct(
    model_q: nn.Module,
    model_fp: nn.Module,
    loader: DataLoader,
    cfg: ReconstructionConfig,
    *,
    granularity: str,
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    """Learn rounding unit by unit (in place on ``model_q``). Returns per-unit loss stats."""
    xs, ys = [], []
    for images, labels in loader:
        xs.append(images)
        ys.append(labels)
        if sum(len(x) for x in xs) >= cfg.num_samples:
            break
    data = torch.cat(xs)[: cfg.num_samples].to(device)
    labels = torch.cat(ys)[: cfg.num_samples].to(device)
    model_q.eval().to(device)
    model_fp.eval().to(device)
    results = {}
    for i, name in enumerate(units(model_q, granularity)):
        unit_q, unit_fp = model_q.get_submodule(name), model_fp.get_submodule(name)
        x_q = _capture(model_q, unit_q, data, cfg.batch_size, want_input=True)
        y_fp = _capture(model_fp, unit_fp, data, cfg.batch_size, want_input=False)
        weight = None
        if cfg.fisher and granularity == "block":
            weight = _fisher(model_fp, unit_fp, data, labels, cfg.batch_size)
            weight = weight / weight.mean().clamp_min(1e-12)
        results[name] = _reconstruct_unit(unit_q, x_q, y_fp, weight, cfg, seed=i)
    return results
