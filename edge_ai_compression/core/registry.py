from __future__ import annotations

from typing import Any

import torch.nn as nn

from edge_ai_compression.models.efficientnet import efficientnet_b0_cifar
from edge_ai_compression.models.mobilenet import mobilenet_v2_cifar
from edge_ai_compression.models.resnet import resnet18_cifar
from edge_ai_compression.models.student_models import small_cnn_student


class ModelRegistry:
    _builders: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, builder: Any) -> None:
        cls._builders[name] = builder

    @classmethod
    def create(cls, name: str, num_classes: int = 10, **kwargs: Any) -> nn.Module:
        if name not in cls._builders:
            raise KeyError(f"Unknown model '{name}'. Registered: {sorted(cls._builders)}")
        return cls._builders[name](num_classes=num_classes, **kwargs)


def _default_register() -> None:
    ModelRegistry.register("resnet18_cifar", resnet18_cifar)
    ModelRegistry.register("mobilenet_v2_cifar", mobilenet_v2_cifar)
    ModelRegistry.register("efficientnet_b0_cifar", efficientnet_b0_cifar)
    ModelRegistry.register("small_cnn_student", small_cnn_student)


_default_register()
