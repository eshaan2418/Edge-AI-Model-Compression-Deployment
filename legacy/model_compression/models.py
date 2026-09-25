from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def create_model(num_classes: int = 10) -> nn.Module:
    """Create a small CNN baseline (ResNet18 adapted for CIFAR-10)."""
    model = tvm.resnet18(weights=None)
    # Adapt for CIFAR-10 small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
