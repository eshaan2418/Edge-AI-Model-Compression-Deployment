from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as tvm


def efficientnet_b0_cifar(num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """EfficientNet-B0 adapted for CIFAR-sized (32x32) inputs.

    The stock stem uses a stride-2 conv; we relax it to stride 1 so that small
    32x32 inputs are not downsampled away before the first block.
    """
    model = tvm.efficientnet_b0(weights=None, num_classes=num_classes)
    first_conv = model.features[0][0]
    first_conv.stride = (1, 1)
    return model
