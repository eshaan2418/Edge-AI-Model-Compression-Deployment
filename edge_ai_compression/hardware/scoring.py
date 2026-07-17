"""Score a model's measured metrics against a hardware profile's budget.

The score blends an accuracy reward with normalized resource-utilization
penalties, plus a hard penalty per constraint violation. Higher is better.
Infeasible candidates (any violated constraint) always score below feasible
ones, so the sign of feasibility is never ambiguous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from edge_ai_compression.hardware.profiles import HardwareProfile, get_profile

# Penalty weights on normalized resource utilization (fraction of budget used).
_W_LATENCY = 0.30
_W_SIZE = 0.30
_W_RAM = 0.20
# Flat penalty added per violated constraint (keeps infeasible < feasible).
_VIOLATION_PENALTY = 1.0

# Accepted metric keys, in priority order, mapped to a canonical name.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "synthetic_accuracy", "acc"),
    "latency_ms": ("latency_ms", "latency_ms_mean", "latency_mean", "latency"),
    "size_mb": ("size_mb", "model_size_mb"),
    "ram_mb": ("ram_mb", "peak_ram_mib", "peak_ram_mb"),
}


@dataclass
class ScoreResult:
    feasible: bool
    score: float
    violations: list[str]
    explanation: str
    utilization: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract(metrics: dict[str, Any], canonical: str) -> float | None:
    for key in _METRIC_ALIASES[canonical]:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def score_candidate(
    metrics: dict[str, Any], hardware_profile: str | HardwareProfile
) -> ScoreResult:
    """Score ``metrics`` against a hardware profile.

    ``metrics`` may use any accepted alias for accuracy / latency / size / RAM
    (e.g. ``latency_ms`` or ``latency_ms_mean``, ``ram_mb`` or ``peak_ram_mib``).
    Missing resource metrics are treated as "unknown" — they cannot violate a
    constraint but also earn no penalty, and are noted in the explanation.
    """
    profile = (
        get_profile(hardware_profile) if isinstance(hardware_profile, str) else hardware_profile
    )

    accuracy = _extract(metrics, "accuracy")
    latency = _extract(metrics, "latency_ms")
    size = _extract(metrics, "size_mb")
    ram = _extract(metrics, "ram_mb")

    violations: list[str] = []
    utilization: dict[str, float] = {}
    penalty = 0.0
    unknown: list[str] = []

    for value, budget, key, weight in (
        (latency, profile.max_latency_ms, "latency_ms", _W_LATENCY),
        (size, profile.max_size_mb, "size_mb", _W_SIZE),
        (ram, profile.max_ram_mb, "ram_mb", _W_RAM),
    ):
        if value is None:
            unknown.append(key)
            continue
        util = value / budget if budget > 0 else float("inf")
        utilization[key] = round(util, 4)
        penalty += weight * util
        if value > budget:
            violations.append(f"{key}={value:g} exceeds {profile.name} budget {budget:g}")

    reward = accuracy if accuracy is not None else 0.0
    if accuracy is None:
        unknown.append("accuracy")

    score = reward - penalty - _VIOLATION_PENALTY * len(violations)
    feasible = not violations

    verdict = "FEASIBLE" if feasible else "INFEASIBLE"
    parts = [
        f"{verdict} on '{profile.name}': score={score:.4f}",
        f"accuracy_reward={reward:.4f}",
        f"resource_penalty={penalty:.4f}",
    ]
    if violations:
        parts.append(f"violations={len(violations)}")
    if unknown:
        parts.append(f"unknown_metrics={sorted(unknown)}")
    explanation = "; ".join(parts)

    return ScoreResult(
        feasible=feasible,
        score=float(score),
        violations=violations,
        explanation=explanation,
        utilization=utilization,
    )
