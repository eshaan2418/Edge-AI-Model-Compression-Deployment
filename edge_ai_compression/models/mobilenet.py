from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as tvm


def mobilenet_v2_cifar(num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """MobileNetV2 adapted for CIFAR-sized (32x32) inputs.

    The first inverted-residual stem uses stride 2, which is aggressive for
    32x32 inputs; we set it to stride 1 to keep enough spatial resolution.
    """
    model = tvm.mobilenet_v2(weights=None, num_classes=num_classes)
    # First conv in features[0] is a ConvNormActivation; relax its stride.
    first_conv = model.features[0][0]
    first_conv.stride = (1, 1)
    return model
