"""Training-run configuration (YAML, strict keys)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

VARIANTS = ("standard",)
SCHEDULES = ("cosine", "constant")


@dataclass(frozen=True)
class TrainConfig:
    """Supervised training with SGD + momentum, linear warmup, then cosine or constant LR.

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
    variant: str = "standard"
    epochs: int = 1
    max_steps: int | None = None
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    warmup_epochs: float = 0.0
    schedule: str = "cosine"
    num_checkpoints: int = 8
    checkpoint_dir: str = "models/checkpoints"
    results_dir: str = "results"
    log_every_steps: int = 50

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant '{self.variant}'; expected {VARIANTS}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule '{self.schedule}'; expected {SCHEDULES}")
        for name in ("batch_size", "epochs", "num_checkpoints", "log_every_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"train.{name} must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("train.max_steps must be >= 1")

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
            if value is None or default is None:
                kwargs[key] = value if value is None else int(value)
            elif isinstance(default, bool):
                kwargs[key] = bool(value)
            else:
                kwargs[key] = type(default)(value)
        return TrainConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
