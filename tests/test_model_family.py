from __future__ import annotations

import pytest
import torch
import torch.fx as fx

from edge_ai_compression.compression.pruning.channel import prune_block_channels
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.models.resnet import RESNET_DEPTHS, RESNET_WIDTHS, resnet_family_name


@pytest.mark.parametrize("depth", list(RESNET_DEPTHS))
@pytest.mark.parametrize("width", RESNET_WIDTHS)
def test_family_forward_and_trace(depth, width):
    m = ModelRegistry.create(resnet_family_name(depth, width), num_classes=100).eval()
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    fx.symbolic_trace(m)


def test_params_grow_with_depth_and_width():
    def n(d, w):
        return sum(p.numel() for p in ModelRegistry.create(resnet_family_name(d, w)).parameters())

    assert n(10, 0.25) < n(10, 0.5) < n(10, 1.0) < n(18, 1.0) < n(34, 1.0)
    # resnet18_w1.0 has the same parameter count as the torchvision-based resnet18_cifar
    ref = sum(p.numel() for p in ModelRegistry.create("resnet18_cifar").parameters())
    assert n(18, 1.0) == ref


def test_family_supports_channel_pruning():
    m = ModelRegistry.create(resnet_family_name(10, 0.5)).eval()
    assert prune_block_channels(m, 0.5) == 4
    assert m(torch.randn(1, 3, 32, 32)).shape == (1, 10)
