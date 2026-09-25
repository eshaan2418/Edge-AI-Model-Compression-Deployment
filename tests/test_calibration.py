from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.compression.quantization.calibration import (
    calibrate_activations,
    calibration_tensor,
)
from edge_ai_compression.compression.quantization.modules import (
    fold_bn,
    quant_layers,
    quantize_model,
)
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.core.registry import ModelRegistry


def _loader(n=64):
    torch.manual_seed(0)
    return DataLoader(
        TensorDataset(torch.randn(n, 3, 32, 32), torch.zeros(n, dtype=torch.long)), batch_size=16
    )


def test_calibration_tensor_stops_at_num_samples():
    assert calibration_tensor(_loader(), 40).shape[0] == 40


@pytest.mark.parametrize("method", ["minmax", "percentile", "mse"])
def test_w8a8_static_close_to_float(method):
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar")
    m.train()
    with torch.no_grad():
        m(torch.randn(16, 3, 32, 32))
    m = fold_bn(m.eval())
    ref = copy.deepcopy(m)
    quantize_model(m, WeightSpec(bits=8), act_bits=8)
    scales = calibrate_activations(m, _loader(), method=method, num_samples=64)
    assert len(scales) == 21 and all(s > 0 for s in scales.values())
    assert all(layer.act_enabled for _, layer in quant_layers(m))
    x = next(iter(_loader()))[0]
    with torch.no_grad():
        err = float((m(x) - ref(x)).norm() / ref(x).norm())
    assert err < 0.1, err


def test_weight_only_layers_are_not_calibrated():
    m = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 4))
    quantize_model(m, WeightSpec(bits=8), act_bits=None, first_last_bits=None)
    assert calibrate_activations(m, _loader(), num_samples=16) == {}
