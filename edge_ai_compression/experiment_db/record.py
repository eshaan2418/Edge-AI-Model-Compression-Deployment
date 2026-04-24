from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

EXPERIMENT_CSV_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "timestamp",
    "model_name",
    "dataset",
    "num_params",
    "flops",
    "compression_order",
    "pruning_type",
    "pruning_sparsity",
    "quantization_type",
    "distillation_enabled",
    "temperature",
    "alpha",
    "device",
    "accuracy",
    "accuracy_drop",
    "latency_mean",
    "latency_std",
    "latency_p50",
    "latency_p90",
    "latency_p99",
    "size_mb",
    "ram_mb",
    "energy_proxy",
)


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
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
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    latency_std: float = 0.0
    size_mb: float = 0.0
    ram_mb: float = 0.0
    energy_proxy: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_csv_row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in EXPERIMENT_CSV_FIELDS}

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "extra": self.extra}


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
    accuracy: float,
    latency: dict[str, float],
    size_mb: float,
    ram_mb: float,
    energy_proxy: float,
    extra: dict[str, Any] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        timestamp=ExperimentRecord.now_ts(),
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
        accuracy=accuracy,
        accuracy_drop=float(baseline_accuracy - accuracy),
        latency_mean=latency["mean"],
        latency_p50=latency["p50"],
        latency_p90=latency["p90"],
        latency_p99=latency["p99"],
        latency_std=latency.get("std", 0.0),
        size_mb=size_mb,
        ram_mb=ram_mb,
        energy_proxy=energy_proxy,
        extra=extra or {},
    )
