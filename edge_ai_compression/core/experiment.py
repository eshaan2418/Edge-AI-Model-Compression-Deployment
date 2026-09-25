from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.core.compression_orders import parse_order
from edge_ai_compression.experiment_db.paths import DEFAULT_RESULTS_DIR


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


QUANT_METHODS = ("rtn", "adaround", "brecq", "qat", "hawq", "smoothquant")
QUANT_KEYS = {
    "enabled",
    "method",
    "weight_bits",
    "act_bits",
    "granularity",
    "group_size",
    "weight_method",
    "first_last_bits",
    "calibration",
    *QUANT_METHODS,
}


@dataclass
class QuantizationSection:
    """PTQ/QAT ladder settings (DECISIONS Phase 3). ``act_bits: null`` = weight-only.

    Method-specific options live under a key named after the method, e.g.
    ``adaround: {iters: 2000}``.
    """

    enabled: bool = False
    method: str = "rtn"
    weight_bits: int = 8
    act_bits: int | None = 8
    granularity: str = "per_channel"
    group_size: int | None = None
    weight_method: str = "minmax"
    first_last_bits: int | None = 8
    calibration: dict[str, Any] = field(
        default_factory=lambda: {"method": "minmax", "num_samples": 512, "percentile": 99.99}
    )
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from edge_ai_compression.compression.quantization.quantizer import WeightSpec, qmax

        # Validate the weight/activation spec at config load, not mid-run.
        WeightSpec(self.weight_bits, self.granularity, self.group_size, self.weight_method)
        if self.act_bits is not None:
            qmax(self.act_bits)

    @property
    def tag(self) -> str:
        act = f"a{self.act_bits}" if self.act_bits else "afp"
        gran = f"g{self.group_size}" if self.group_size else self.granularity
        bits = (
            f"avg{self.options.get('avg_bits', 6.0)}" if self.method == "hawq" else self.weight_bits
        )
        return f"{self.method}:w{bits}{act}:{gran}"

    def to_dict(self) -> dict[str, Any]:
        """YAML shape (method options under the method's key); round-trips with from_dict."""
        d = asdict(self)
        options = d.pop("options")
        if options:
            d[self.method] = options
        return d

    @staticmethod
    def from_dict(q: dict[str, Any]) -> QuantizationSection:
        if "mode" in q:
            raise ValueError(
                "quantization.mode was replaced by quantization.method; the torch.ao "
                "dynamic_linear path was removed (DECISIONS D3.2)"
            )
        unknown = set(q) - QUANT_KEYS
        if unknown:
            raise ValueError(f"unknown quantization keys: {sorted(unknown)}")
        method = str(q.get("method", "rtn"))
        if method not in QUANT_METHODS:
            raise ValueError(f"unknown quantization method '{method}'; expected {QUANT_METHODS}")
        calibration = {"method": "minmax", "num_samples": 512, "percentile": 99.99}
        calibration.update(q.get("calibration") or {})
        act_bits = q.get("act_bits", 8)
        group = q.get("group_size")
        flb = q.get("first_last_bits", 8)
        return QuantizationSection(
            enabled=bool(q.get("enabled", False)),
            method=method,
            weight_bits=int(q.get("weight_bits", 8)),
            act_bits=int(act_bits) if act_bits is not None else None,
            granularity=str(q.get("granularity", "per_group" if group else "per_channel")),
            group_size=int(group) if group is not None else None,
            weight_method=str(q.get("weight_method", "minmax")),
            first_last_bits=int(flb) if flb is not None else None,
            calibration=calibration,
            options=dict(q.get(method) or {}),
        )


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
            quantization=QuantizationSection.from_dict(q),
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


@dataclass(frozen=True)
class ExperimentDBConfig:
    """The ``experiment_db:`` section. The DB is always on; it is the only results output."""

    results_dir: str = str(DEFAULT_RESULTS_DIR)
    diagnostics_max_batches: int = 15

    @property
    def path(self) -> Path:
        return Path(self.results_dir)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExperimentDBConfig:
        if "enabled" in d:
            raise ValueError("experiment_db.enabled was removed: the experiment DB is always on")
        unknown = set(d) - {"results_dir", "diagnostics_max_batches"}
        if unknown:
            raise ValueError(f"unknown experiment_db keys: {sorted(unknown)}")
        return ExperimentDBConfig(
            results_dir=str(d.get("results_dir", DEFAULT_RESULTS_DIR)),
            diagnostics_max_batches=int(d.get("diagnostics_max_batches", 15)),
        )


def dataset_num_classes(name: str) -> int:
    n = name.lower()
    if n in ("cifar10", "fake", "synthetic", "debug", "random"):
        return 10
    if n == "cifar100":
        return 100
    if n in ("tiny_imagenet", "tiny-imagenet-200"):
        return 200
    if n == "imagenet":
        return 1000
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
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    train: dict[str, Any] = field(default_factory=dict)
    experiment_db: ExperimentDBConfig = field(default_factory=ExperimentDBConfig)

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
            benchmark=BenchmarkConfig.from_dict(dict(d.get("benchmark") or {})),
            train=dict(d.get("train") or {}),
            experiment_db=ExperimentDBConfig.from_dict(dict(d.get("experiment_db") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-YAML view of the resolved config (stored with every run)."""
        d = asdict(self)
        d["benchmark"] = self.benchmark.to_dict()
        d["compression"]["quantization"] = self.compression.quantization.to_dict()
        return d


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
