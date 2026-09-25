"""Aggregate per-process benchmark results into one report."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.fingerprint import Fingerprint
from edge_ai_compression.benchmarking.isolation import ProcessResult
from edge_ai_compression.benchmarking.stats import Summary, bootstrap_ci, summarize


@dataclass(frozen=True)
class BenchmarkReport:
    """Accuracy plus latency/memory measured over ``process_repeats`` fresh processes.

    ``latency`` summarizes all timed iterations pooled across processes
    (descriptive only). ``latency_median_ms`` is the median of per-process
    medians and ``latency_median_ci_ms`` its bootstrap CI over processes
    (``None`` with a single process); use these for comparisons (DECISIONS D1.1).
    """

    accuracy: float
    val_loss: float
    input_shape: tuple[int, ...]
    config: BenchmarkConfig
    latency: Summary
    process_medians_ms: tuple[float, ...]
    latency_median_ms: float
    latency_median_ci_ms: tuple[float, float] | None
    stages_ms: dict[str, float]
    peak_rss_mib: float
    model_peak_rss_mib: float
    size_mb: float
    energy_j_per_inf: float | None
    fingerprint: Fingerprint
    environment_problems: list[str]
    traces_ns: tuple[np.ndarray, ...] = field(repr=False)

    @property
    def cold_start_ms(self) -> float:
        return self.stages_ms["cold_start"]

    @staticmethod
    def from_runs(
        runs: Sequence[ProcessResult],
        *,
        accuracy: float,
        val_loss: float,
        input_shape: tuple[int, ...],
        config: BenchmarkConfig,
        size_mb: float,
        fingerprint: Fingerprint,
        environment_problems: list[str],
    ) -> BenchmarkReport:
        if not runs:
            raise ValueError("no benchmark runs")
        medians = tuple(float(np.median(r.trace_ms)) for r in runs)
        stages = {k: float(np.median([r.stages_ms[k] for r in runs])) for k in runs[0].stages_ms}
        energies = [r.energy_j_per_inf for r in runs if r.energy_j_per_inf is not None]
        return BenchmarkReport(
            accuracy=accuracy,
            val_loss=val_loss,
            input_shape=input_shape,
            config=config,
            latency=summarize(np.concatenate([r.trace_ms for r in runs])),
            process_medians_ms=medians,
            latency_median_ms=float(np.median(medians)),
            latency_median_ci_ms=bootstrap_ci(medians) if len(medians) > 1 else None,
            stages_ms=stages,
            peak_rss_mib=float(np.median([r.peak_rss_mib for r in runs])),
            model_peak_rss_mib=float(np.median([r.model_peak_rss_mib for r in runs])),
            size_mb=size_mb,
            energy_j_per_inf=float(np.median(energies)) if energies else None,
            fingerprint=fingerprint,
            environment_problems=environment_problems,
            traces_ns=tuple(r.trace_ns for r in runs),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (traces excluded; they are stored as an artifact)."""
        return {
            "accuracy": self.accuracy,
            "val_loss": self.val_loss,
            "input_shape": list(self.input_shape),
            "config": self.config.to_dict(),
            "latency_ms": self.latency.to_dict(),
            "process_medians_ms": list(self.process_medians_ms),
            "latency_median_ms": self.latency_median_ms,
            "latency_median_ci_ms": (
                list(self.latency_median_ci_ms) if self.latency_median_ci_ms else None
            ),
            "stages_ms": self.stages_ms,
            "peak_rss_mib": self.peak_rss_mib,
            "model_peak_rss_mib": self.model_peak_rss_mib,
            "size_mb": self.size_mb,
            "energy_j_per_inf": self.energy_j_per_inf,
            "fingerprint_hash": self.fingerprint.hash,
            "environment_problems": self.environment_problems,
        }
