"""The ``benchmark:`` section of an experiment config."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from edge_ai_compression.benchmarking.energy import METERS

RENAMED_KEYS = {
    "latency_repeats": "iters",
    "latency_warmup": "warmup_iters",
    "results_md": None,
    "results_jsonl": None,
    "input_shape": "batch_size (spatial dims come from the dataset)",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """How latency and memory are measured.

    The benchmark input is ``(batch_size, *sample_shape)`` where the sample shape
    comes from the dataset (or ``--input-shape`` in the standalone CLI).

    Each of ``process_repeats`` fresh processes runs at least ``warmup_iters``
    untimed iterations (and for at least ``min_warmup_s``), then at least
    ``iters`` timed iterations (and for at least ``min_time_s``).
    """

    batch_size: int = 1
    num_threads: int = 4
    warmup_iters: int = 50
    min_warmup_s: float = 1.0
    iters: int = 1000
    min_time_s: float = 0.0
    process_repeats: int = 5
    cpu_affinity: tuple[int, ...] | None = None
    energy_meter: str = "none"
    strict_environment: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "num_threads", "iters", "process_repeats"):
            if getattr(self, name) < 1:
                raise ValueError(f"benchmark.{name} must be >= 1")
        for name in ("warmup_iters", "min_warmup_s", "min_time_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"benchmark.{name} must be >= 0")
        if self.energy_meter not in METERS:
            raise ValueError(
                f"unknown benchmark.energy_meter '{self.energy_meter}'; "
                f"implemented: {sorted(METERS)}"
            )
        if self.cpu_affinity is not None and not self.cpu_affinity:
            raise ValueError("benchmark.cpu_affinity must be omitted or non-empty")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> BenchmarkConfig:
        known = {f.name for f in fields(BenchmarkConfig)}
        for key in d:
            if key in known:
                continue
            if key in RENAMED_KEYS:
                new = RENAMED_KEYS[key]
                hint = f"renamed to '{new}'" if new else "removed (the experiment DB is the output)"
                raise ValueError(f"benchmark.{key} was {hint}")
            raise ValueError(f"unknown benchmark key '{key}'; expected one of {sorted(known)}")
        kwargs: dict[str, Any] = dict(d)
        if kwargs.get("cpu_affinity") is not None:
            kwargs["cpu_affinity"] = tuple(int(x) for x in kwargs["cpu_affinity"])
        for name in ("batch_size", "num_threads", "warmup_iters", "iters", "process_repeats"):
            if name in kwargs:
                kwargs[name] = int(kwargs[name])
        for name in ("min_warmup_s", "min_time_s"):
            if name in kwargs:
                kwargs[name] = float(kwargs[name])
        if "energy_meter" in kwargs:
            kwargs["energy_meter"] = str(kwargs["energy_meter"])
        if "strict_environment" in kwargs:
            kwargs["strict_environment"] = bool(kwargs["strict_environment"])
        return BenchmarkConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cpu_affinity"] = list(self.cpu_affinity) if self.cpu_affinity else None
        return d
