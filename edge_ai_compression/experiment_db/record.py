from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from edge_ai_compression.benchmarking.report import BenchmarkReport

SCHEMA_VERSION = 2


@dataclass
class ExperimentRecord:
    """One row of ``experiments.csv``. Latency columns are milliseconds.

    ``latency_median`` (+ CI) is the median of per-process medians and is the
    number to compare across runs. ``latency_mean/std/p*/max`` describe all timed
    iterations pooled across processes.
    """

    experiment_id: str
    timestamp: str
    schema_version: int
    git_commit: str | None
    git_dirty: bool | None
    fingerprint_hash: str
    model_name: str
    dataset: str
    num_params: int
    flops: float
    compression_order: str
    pruning_type: str
    pruning_sparsity: float
    quantization_type: str
    distillation_enabled: bool
    temperature: float
    alpha: float
    device: str
    accuracy: float
    accuracy_drop: float
    latency_median: float
    latency_median_ci_lo: float | None
    latency_median_ci_hi: float | None
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    n_iters: int
    process_repeats: int
    num_threads: int
    batch_size: int
    cold_start_ms: float
    peak_rss_mib: float
    model_peak_rss_mib: float
    size_mb: float
    energy_j_per_inf: float | None
    energy_meter: str
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_csv_row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in EXPERIMENT_CSV_FIELDS}

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


EXPERIMENT_CSV_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ExperimentRecord) if f.name != "extra"
)


def record_from_run(
    *,
    experiment_id: str,
    model_name: str,
    dataset: str,
    num_params: int,
    flops: float,
    compression_order: str,
    pruning_type: str,
    pruning_sparsity: float,
    quantization_type: str,
    distillation_enabled: bool,
    temperature: float,
    alpha: float,
    device: str,
    baseline_accuracy: float,
    report: BenchmarkReport,
    extra: dict[str, Any] | None = None,
) -> ExperimentRecord:
    lat = report.latency
    ci = report.latency_median_ci_ms
    cfg = report.config
    return ExperimentRecord(
        experiment_id=experiment_id,
        timestamp=ExperimentRecord.now_ts(),
        schema_version=SCHEMA_VERSION,
        git_commit=report.fingerprint.git_commit,
        git_dirty=report.fingerprint.git_dirty,
        fingerprint_hash=report.fingerprint.hash,
        model_name=model_name,
        dataset=dataset,
        num_params=num_params,
        flops=flops,
        compression_order=compression_order,
        pruning_type=pruning_type,
        pruning_sparsity=pruning_sparsity,
        quantization_type=quantization_type,
        distillation_enabled=distillation_enabled,
        temperature=temperature,
        alpha=alpha,
        device=device,
        accuracy=report.accuracy,
        accuracy_drop=float(baseline_accuracy - report.accuracy),
        latency_median=report.latency_median_ms,
        latency_median_ci_lo=ci[0] if ci else None,
        latency_median_ci_hi=ci[1] if ci else None,
        latency_mean=lat.mean,
        latency_std=lat.std,
        latency_p50=lat.p50,
        latency_p90=lat.p90,
        latency_p95=lat.p95,
        latency_p99=lat.p99,
        latency_max=lat.max,
        n_iters=lat.n,
        process_repeats=cfg.process_repeats,
        num_threads=cfg.num_threads,
        batch_size=cfg.batch_size,
        cold_start_ms=report.cold_start_ms,
        peak_rss_mib=report.peak_rss_mib,
        model_peak_rss_mib=report.model_peak_rss_mib,
        size_mb=report.size_mb,
        energy_j_per_inf=report.energy_j_per_inf,
        energy_meter=cfg.energy_meter,
        extra=extra or {},
    )
