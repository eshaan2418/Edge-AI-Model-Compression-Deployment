"""Training-run configuration (YAML, strict keys)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEDULES = ("cosine", "constant")
OPTIMIZERS = ("sgd", "adamw")


@dataclass(frozen=True)
class TrainConfig:
    """Supervised training (SGD + nesterov momentum, or AdamW), linear warmup, then
    cosine or constant LR.

    ``num_checkpoints`` checkpoints are saved at log-spaced steps (always
    including the final step) so downstream compression can start from any
    point in training.
    """

    model: str = "resnet18_cifar"
    dataset: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    limit_samples: int | None = None
    device: str = "cpu"
    seed: int = 0
    variant: str = "standard"  # standard | quant_noise | kurtosis | rigl
    variant_options: dict[str, Any] = field(default_factory=dict)
    epochs: int = 1
    max_steps: int | None = None
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    warmup_epochs: float = 0.0
    schedule: str = "cosine"
    num_checkpoints: int = 8
    checkpoint_dir: str = "models/checkpoints"
    export_path: str | None = None  # also copy the final checkpoint here (stable path for configs)
    results_dir: str = "results"
    log_every_steps: int = 50
    signals: dict[str, Any] | None = None  # SignalConfig options; None disables signal logging
    signal_every_steps: int | None = None  # in addition to every checkpoint step

    def __post_init__(self) -> None:
        from edge_ai_compression.pretraining.variants import make_variant

        make_variant(self.variant, self.variant_options)  # validates name and options
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer '{self.optimizer}'; expected {OPTIMIZERS}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule '{self.schedule}'; expected {SCHEDULES}")
        for name in ("batch_size", "epochs", "num_checkpoints", "log_every_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"train.{name} must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("train.max_steps must be >= 1")
        if self.signals is not None:
            from edge_ai_compression.pretraining.signals import SignalConfig

            SignalConfig.from_dict(self.signals)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TrainConfig:
        known = {f.name: f for f in fields(TrainConfig)}
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"unknown train keys: {sorted(unknown)}")
        defaults = TrainConfig()
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            default = getattr(defaults, key)
            if key == "export_path":
                kwargs[key] = None if value is None else str(value)
            elif key == "signals":
                kwargs[key] = None if value is None else dict(value)
            elif key == "variant_options":
                kwargs[key] = dict(value or {})
            elif value is None or default is None:
                kwargs[key] = value if value is None else int(value)
            elif isinstance(default, bool):
                kwargs[key] = bool(value)
            else:
                kwargs[key] = type(default)(value)
        return TrainConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
