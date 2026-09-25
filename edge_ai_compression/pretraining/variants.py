"""Compression-aware training variants (DECISIONS Phase 5).

A variant hooks into the training loop:
    setup(model, total_steps) -> extra_loss(model) -> after_backward(model, step)
    -> [optimizer.step] -> after_step(model, optimizer, step) ... finalize(model)

- ``quant_noise``: Quant-Noise (Fan et al., ICLR 2021). Each training forward
  fake-quantizes a random fraction ``p`` of each conv/linear weight to ``bits``
  bits (per output channel), straight-through. ``p = 1`` is QAT from scratch.
  Inactive in eval mode, so signals and checkpoints see clean float weights.
- ``kurtosis``: adds lam * sum_l (Kurt(W_l) - target)^2 (Shkolnik et al.,
  NeurIPS 2020; target 1.8 = uniform).
- ``rigl``: RigL (Evci et al., ICML 2020). ERK (or uniform) per-layer
  densities, random sparse init; every ``delta_t`` steps until ``t_end`` of
  training, drop the ``f(t)`` fraction of smallest-magnitude active weights and
  grow as many inactive weights with the largest dense-gradient magnitude,
  f(t) = alpha/2 * (1 + cos(pi t / t_end)). Masks enforced by re-zeroing
  weights and optimizer state after each step (D5.1).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from edge_ai_compression.compression.quantization.quantizer import fake_quant, qmax

VARIANT_OPTIONS = {
    "standard": {},
    "quant_noise": {"bits": 4, "p": 0.5},
    "kurtosis": {"lam": 1.0, "target": 1.8},
    "rigl": {
        "sparsity": 0.9,
        "distribution": "erk",
        "delta_t": 100,
        "alpha": 0.3,
        "t_end": 0.75,
        "dense_first_layer": True,
        "seed": 0,
    },
}


def _layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d | nn.Linear)]


def _options(name: str, given: dict[str, Any]) -> dict[str, Any]:
    defaults = VARIANT_OPTIONS[name]
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {name} options: {sorted(unknown)}")
    return {**defaults, **given}


class Variant:
    def setup(self, model: nn.Module, total_steps: int) -> None: ...

    def extra_loss(self, model: nn.Module) -> torch.Tensor | float:
        return 0.0

    def after_backward(self, model: nn.Module, step: int) -> None: ...

    def after_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None: ...

    def finalize(self, model: nn.Module) -> None: ...


class _QuantNoiseParam(nn.Module):
    def __init__(self, bits: int, p: float) -> None:
        super().__init__()
        self.bits, self.p = bits, p

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return w
        w2d = w.reshape(w.shape[0], -1)
        scale = (w2d.detach().abs().amax(dim=1, keepdim=True) / qmax(self.bits)).clamp_min(1e-12)
        q = fake_quant(w2d, scale, self.bits).reshape(w.shape)
        mask = (torch.rand_like(w) < self.p).to(w.dtype)
        return w + (mask * (q - w)).detach()


class QuantNoise(Variant):
    def __init__(self, bits: int, p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("quant_noise p must be in [0, 1]")
        self.bits, self.p = int(bits), float(p)

    def setup(self, model: nn.Module, total_steps: int) -> None:
        for _, mod in _layers(model):
            parametrize.register_parametrization(mod, "weight", _QuantNoiseParam(self.bits, self.p))

    def finalize(self, model: nn.Module) -> None:
        for _, mod in _layers(model):
            if parametrize.is_parametrized(mod, "weight"):
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)


def differentiable_kurtosis(w: torch.Tensor) -> torch.Tensor:
    c = w.reshape(-1) - w.mean()
    var = c.pow(2).mean()
    return c.pow(4).mean() / var.pow(2).clamp_min(1e-24)


class Kurtosis(Variant):
    def __init__(self, lam: float, target: float) -> None:
        self.lam, self.target = float(lam), float(target)

    def extra_loss(self, model: nn.Module) -> torch.Tensor | float:
        terms = [(differentiable_kurtosis(m.weight) - self.target) ** 2 for _, m in _layers(model)]
        return self.lam * torch.stack(terms).sum()


def erk_densities(
    shapes: dict[str, tuple[int, ...]], sparsity: float, dense: set[str]
) -> dict[str, float]:
    """Erdos-Renyi-Kernel densities averaging to 1 - sparsity over all weights.

    Layer density is proportional to sum(shape) / prod(shape), capped at 1; capped
    and ``dense`` layers are fixed and the rest re-solved (Evci et al. 2020).
    """
    sizes = {n: math.prod(s) for n, s in shapes.items()}
    budget = (1.0 - sparsity) * sum(sizes.values())
    fixed = {n: 1.0 for n in dense}
    while True:
        free = [n for n in shapes if n not in fixed]
        remaining = budget - sum(sizes[n] for n in fixed)
        raw = {n: sum(shapes[n]) / sizes[n] for n in free}
        denom = sum(raw[n] * sizes[n] for n in free)
        eps = 0.0 if not free or remaining <= 0 else remaining / denom
        over = [n for n in free if eps * raw[n] > 1.0]
        if not over:
            return {**fixed, **{n: eps * raw[n] for n in free}}
        fixed.update({n: 1.0 for n in over})


class RigL(Variant):
    def __init__(
        self,
        sparsity: float,
        distribution: str,
        delta_t: int,
        alpha: float,
        t_end: float,
        dense_first_layer: bool,
        seed: int,
    ) -> None:
        if distribution not in ("erk", "uniform"):
            raise ValueError("rigl distribution must be erk or uniform")
        if not 0.0 < sparsity < 1.0:
            raise ValueError("rigl sparsity must be in (0, 1)")
        self.sparsity, self.distribution = float(sparsity), distribution
        self.delta_t, self.alpha, self.t_end = int(delta_t), float(alpha), float(t_end)
        self.dense_first_layer, self.seed = bool(dense_first_layer), int(seed)
        self.masks: dict[str, torch.Tensor] = {}
        self.grads: dict[str, torch.Tensor] = {}
        self.total = 1

    def setup(self, model: nn.Module, total_steps: int) -> None:
        self.total = max(total_steps, 1)
        layers = _layers(model)
        dense = {layers[0][0]} if self.dense_first_layer and layers else set()
        shapes = {n: tuple(m.weight.shape) for n, m in layers}
        if self.distribution == "erk":
            dens = erk_densities(shapes, self.sparsity, dense)
        else:
            dens = {n: (1.0 if n in dense else 1.0 - self.sparsity) for n in shapes}
        gen = torch.Generator().manual_seed(self.seed)
        for name, mod in layers:
            n = mod.weight.numel()
            keep = int(round(dens[name] * n))
            idx = torch.randperm(n, generator=gen)[:keep]
            mask = torch.zeros(n)
            mask[idx] = 1.0
            self.masks[name] = mask.reshape(mod.weight.shape).to(mod.weight.device)
            with torch.no_grad():
                mod.weight.mul_(self.masks[name])

    def update_fraction(self, step: int) -> float:
        return self.alpha / 2 * (1 + math.cos(math.pi * step / (self.t_end * self.total)))

    def _is_update(self, step: int) -> bool:
        return step > 0 and step % self.delta_t == 0 and step < self.t_end * self.total

    def after_backward(self, model: nn.Module, step: int) -> None:
        if not self._is_update(step):
            return
        frac = self.update_fraction(step)
        for name, mod in _layers(model):
            mask, grad = self.masks[name], mod.weight.grad
            if grad is None or bool(mask.all()):
                continue
            flat_m = mask.reshape(-1)
            n_active = int(flat_m.sum())
            k = int(frac * n_active)
            if k == 0:
                continue
            w = mod.weight.detach().abs().reshape(-1)
            drop_scores = torch.where(flat_m.bool(), w, torch.full_like(w, float("inf")))
            drop = drop_scores.topk(k, largest=False).indices
            new_m = flat_m.clone()
            new_m[drop] = 0.0
            g = grad.detach().abs().reshape(-1)
            grow_scores = torch.where(new_m.bool(), torch.full_like(g, -float("inf")), g)
            grow = grow_scores.topk(k).indices
            new_m[grow] = 1.0
            self.masks[name] = new_m.reshape(mask.shape)
            with torch.no_grad():  # grown connections start at zero
                mod.weight.reshape(-1)[grow] = 0.0

    def after_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
        with torch.no_grad():
            for name, mod in _layers(model):
                mask = self.masks[name]
                mod.weight.mul_(mask)
                state = optimizer.state.get(mod.weight, {})
                for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                    if state.get(key) is not None:
                        state[key].mul_(mask)

    def density(self) -> float:
        total = sum(m.numel() for m in self.masks.values())
        return sum(float(m.sum()) for m in self.masks.values()) / max(total, 1)


def make_variant(name: str, options: dict[str, Any]) -> Variant:
    if name not in VARIANT_OPTIONS:
        raise ValueError(f"unknown variant '{name}'; expected {sorted(VARIANT_OPTIONS)}")
    opts = _options(name, options)
    if name == "quant_noise":
        return QuantNoise(**opts)
    if name == "kurtosis":
        return Kurtosis(**opts)
    if name == "rigl":
        return RigL(**opts)
    return Variant()
