from __future__ import annotations

from typing import Any

import torch.nn as nn

from edge_ai_compression.models.efficientnet import efficientnet_b0_cifar
from edge_ai_compression.models.mobilenet import mobilenet_v2_cifar
from edge_ai_compression.models.resnet import (
    RESNET_DEPTHS,
    RESNET_WIDTHS,
    resnet18_cifar,
    resnet_cifar,
    resnet_family_name,
)
from edge_ai_compression.models.student_models import small_cnn_student
from edge_ai_compression.models.vit import VIT_SIZES, vit_cifar


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


def _torchvision_imagenet(arch: str):
    """Pretrained torchvision ImageNet model (downloads weights on first use)."""

    def build(num_classes: int = 1000) -> nn.Module:
        import torchvision

        if num_classes != 1000:
            raise ValueError(f"{arch}_imagenet_tv is a 1000-class ImageNet model")
        weights = torchvision.models.get_model_weights(arch).IMAGENET1K_V1
        return torchvision.models.get_model(arch, weights=weights)

    return build


def _default_register() -> None:
    ModelRegistry.register("resnet18_cifar", resnet18_cifar)
    ModelRegistry.register("mobilenet_v2_cifar", mobilenet_v2_cifar)
    ModelRegistry.register("efficientnet_b0_cifar", efficientnet_b0_cifar)
    ModelRegistry.register("small_cnn_student", small_cnn_student)
    for arch in ("resnet18", "resnet50"):
        ModelRegistry.register(f"{arch}_imagenet_tv", _torchvision_imagenet(arch))
    for depth in RESNET_DEPTHS:
        for width in RESNET_WIDTHS:
            ModelRegistry.register(
                resnet_family_name(depth, width),
                lambda num_classes=10, _d=depth, _w=width: resnet_cifar(_d, _w, num_classes),
            )
    for size in VIT_SIZES:
        ModelRegistry.register(size, lambda num_classes=10, _s=size: vit_cifar(_s, num_classes))


_default_register()
