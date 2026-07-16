from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as tvm


def resnet18_cifar(num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """ResNet-18 adapted for CIFAR-sized (32x32) inputs.

    The stock ImageNet stem (7x7 stride-2 conv + maxpool) throws away too much
    spatial resolution for 32x32 images, so we replace it with a 3x3 stride-1
    conv and drop the initial maxpool, following common CIFAR practice.
    """
    model = tvm.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
