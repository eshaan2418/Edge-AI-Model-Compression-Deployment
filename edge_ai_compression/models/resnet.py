from __future__ import annotations

from typing import Any

import torch
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


# Scaling family for Phase 5: BasicBlock ResNets with a CIFAR stem, varied in
# depth (blocks per stage) and width (multiplier on 64/128/256/512 channels).
RESNET_DEPTHS = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}
RESNET_WIDTHS = (0.25, 0.5, 1.0)


class ResNetCIFAR(nn.Module):
    """torchvision BasicBlocks and attribute names (conv1/bn1/layer1-4/fc), so
    channel pruning, BRECQ block discovery, and the engine's FX lowering apply."""

    def __init__(self, blocks: list[int], width: float, num_classes: int = 10) -> None:
        super().__init__()
        planes = [max(8, int(round(c * width))) for c in (64, 128, 256, 512)]
        self.conv1 = nn.Conv2d(3, planes[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.ReLU(inplace=True)
        inplanes = planes[0]
        stages = []
        for i, (n, p) in enumerate(zip(blocks, planes, strict=True)):
            stride = 1 if i == 0 else 2
            layers: list[nn.Module] = []
            for j in range(n):
                s = stride if j == 0 else 1
                down = None
                if s != 1 or inplanes != p:
                    down = nn.Sequential(
                        nn.Conv2d(inplanes, p, 1, s, bias=False), nn.BatchNorm2d(p)
                    )
                layers.append(tvm.resnet.BasicBlock(inplanes, p, s, down))
                inplanes = p
            stages.append(nn.Sequential(*layers))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def resnet_family_name(depth: int, width: float) -> str:
    return f"resnet{depth}_w{width}_cifar"


def resnet_cifar(depth: int, width: float, num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(RESNET_DEPTHS[depth], width, num_classes)
