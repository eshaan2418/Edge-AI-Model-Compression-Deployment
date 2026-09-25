"""Energy measurement interface (DECISIONS D1.9).

Only ``NullMeter`` is implemented. A real sampler (powermetrics on macOS, RAPL
on x86 Linux, a USB meter) implements ``EnergyMeter``, registers in ``METERS``,
and fills the existing ``energy_j_per_inf`` column: no schema change needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class EnergyMeter(Protocol):
    name: str

    def joules_per_call(self, fn: Callable[[], None], duration_s: float) -> float | None:
        """Run ``fn`` repeatedly for ``duration_s`` and return mean joules per call."""
        ...


class NullMeter:
    name = "none"

    def joules_per_call(self, fn: Callable[[], None], duration_s: float) -> float | None:
        return None


METERS: dict[str, type[EnergyMeter]] = {"none": NullMeter}


def get_meter(name: str) -> EnergyMeter:
    if name not in METERS:
        raise ValueError(f"unknown energy_meter '{name}'; implemented: {sorted(METERS)}")
    return METERS[name]()
