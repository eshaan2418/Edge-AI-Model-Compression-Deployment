from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyEstimate:
    """Rough first-order energy proxy (relative units), not Joules without hardware counters."""

    score: float
    notes: str


def estimate_energy_proxy(latency_ms_mean: float, model_size_mb: float) -> EnergyEstimate:
    """Heuristic: energy grows with time-on-device and memory traffic (size proxy)."""
    score = latency_ms_mean * 0.6 + model_size_mb * 0.4
    return EnergyEstimate(
        score=float(score),
        notes=(
            "Unitless proxy; calibrate with on-device power measurement for research-grade numbers."
        ),
    )
