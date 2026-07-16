from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edge_ai_compression.core.compression_orders import parse_order


def _section(d: dict[str, Any], key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    v = d.get(key)
    if isinstance(v, dict):
        return v
    return dict(default or {})


@dataclass
class PruningSection:
    enabled: bool = False
    amount: float = 0.5
    mode: str = "global_unstructured"
    scorer: str = "magnitude"


@dataclass
class QuantizationSection:
    enabled: bool = False
    mode: str = "dynamic_linear"


@dataclass
class DistillationSection:
    enabled: bool = False
    teacher_ckpt: str = "models/baseline_resnet18.pt"
    teacher_model: str = "resnet18_cifar"
    temperature: float = 4.0
    alpha: float = 0.5
    epochs: int = 1
    lr: float = 0.05


@dataclass
class CompressionConfig:
    pruning: PruningSection = field(default_factory=PruningSection)
    quantization: QuantizationSection = field(default_factory=QuantizationSection)
    distillation: DistillationSection = field(default_factory=DistillationSection)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CompressionConfig:
        p = _section(d, "pruning")
        q = _section(d, "quantization")
        di = _section(d, "distillation")
        return CompressionConfig(
            pruning=PruningSection(
                enabled=bool(p.get("enabled", False)),
                amount=float(p.get("amount", 0.5)),
                mode=str(p.get("mode", "global_unstructured")),
                scorer=str(p.get("scorer", "magnitude")),
            ),
            quantization=QuantizationSection(
                enabled=bool(q.get("enabled", False)),
                mode=str(q.get("mode", "dynamic_linear")),
            ),
            distillation=DistillationSection(
                enabled=bool(di.get("enabled", False)),
                teacher_ckpt=str(di.get("teacher_ckpt", "models/baseline_resnet18.pt")),
                teacher_model=str(di.get("teacher_model", "resnet18_cifar")),
                temperature=float(di.get("temperature", 4.0)),
                alpha=float(di.get("alpha", 0.5)),
                epochs=int(di.get("epochs", 1)),
                lr=float(di.get("lr", 0.05)),
            ),
        )


def dataset_num_classes(name: str) -> int:
    n = name.lower()
    if n in ("cifar10", "fake", "synthetic", "debug", "random"):
        return 10
    if n == "cifar100":
        return 100
    if n in ("tiny_imagenet", "tiny-imagenet-200"):
        return 200
    return 10


@dataclass
class ExperimentConfig:
    model: str = "resnet18_cifar"
    dataset: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cpu"
    seed: int = 42
    limit_samples: int | None = None
    checkpoint_in: str | None = None
    checkpoint_out: str = "models/compressed.pt"
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    compression_order: list[str] = field(default_factory=lambda: ["distill", "prune", "quantize"])
    hardware_profile: str = "laptop_cpu"
    benchmark: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    experiment_db: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return dataset_num_classes(self.dataset)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExperimentConfig:
        comp = CompressionConfig.from_dict(_section(d, "compression"))
        order: list[str]
        if d.get("compression_order_tag"):
            order = parse_order(str(d["compression_order_tag"]))
        elif d.get("compression_order"):
            raw = d["compression_order"]
            order = parse_order(raw) if isinstance(raw, str) else [str(x) for x in raw]
        else:
            order = ["distill", "prune", "quantize"]
        return ExperimentConfig(
            model=str(d.get("model", "resnet18_cifar")),
            dataset=str(d.get("dataset", "cifar10")),
            data_dir=str(d.get("data_dir", "data")),
            batch_size=int(d.get("batch_size", 128)),
            num_workers=int(d.get("num_workers", 2)),
            device=str(d.get("device", "cpu")),
            seed=int(d.get("seed", 42)),
            limit_samples=(int(d["limit_samples"]) if d.get("limit_samples") is not None else None),
            checkpoint_in=d.get("checkpoint_in"),
            checkpoint_out=str(d.get("checkpoint_out", "models/compressed.pt")),
            compression=comp,
            compression_order=order,
            hardware_profile=str(d.get("hardware_profile", "laptop_cpu")),
            benchmark=dict(d.get("benchmark") or {}),
            train=dict(d.get("train") or {}),
            experiment_db=dict(d.get("experiment_db") or {}),
        )


@dataclass
class ExperimentResult:
    accuracy: float
    latency_ms_mean: float
    latency_ms_p99: float
    size_mb: float
    peak_ram_mib: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_p99": self.latency_ms_p99,
            "size_mb": self.size_mb,
            "peak_ram_mib": self.peak_ram_mib,
            **self.extras,
        }
