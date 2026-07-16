from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from typing import Any


def distillation_hyperparameter_grid(
    temperatures: Iterable[float],
    alphas: Iterable[float],
) -> list[dict[str, float]]:
    """Phase 11 — enumerate (temperature, alpha) pairs for distillation sweeps."""
    return [{"temperature": float(t), "alpha": float(a)} for t, a in product(temperatures, alphas)]


def run_sweep(
    configs: list[dict[str, float]],
    runner: Callable[[dict[str, float]], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [{**cfg, **runner(cfg)} for cfg in configs]
