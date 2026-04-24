from __future__ import annotations

from typing import Any


def scalarized_objective(
    result: dict[str, Any],
    w_accuracy: float = 1.0,
    w_latency: float = 0.01,
    w_size: float = 0.02,
    w_ram: float = 0.005,
) -> float:
    return (
        w_accuracy * float(result["accuracy"])
        - w_latency * float(result["latency_ms_mean"])
        - w_size * float(result["size_mb"])
        - w_ram * float(result["peak_ram_mib"])
    )
