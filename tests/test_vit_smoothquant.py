from __future__ import annotations

import copy

import pytest
import torch
import torch.fx as fx
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.compression.quantization.calibration import calibrate_activations
from edge_ai_compression.compression.quantization.modules import quantize_model
from edge_ai_compression.compression.quantization.ptq import run_quantization
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.compression.quantization.smoothquant import (
    SmoothQuantConfig,
    norm_linear_groups,
    outlier_stats,
    smooth,
)
from edge_ai_compression.core.experiment import QuantizationSection
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.models.vit import VIT_SIZES


def _loader(n=32):
    torch.manual_seed(1)
    return DataLoader(
        TensorDataset(torch.randn(n, 3, 32, 32), torch.zeros(n, dtype=torch.long)), batch_size=16
    )


@pytest.mark.parametrize("name", list(VIT_SIZES))
def test_vit_sizes_forward_and_trace(name):
    m = ModelRegistry.create(name, num_classes=100).eval()
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    fx.symbolic_trace(m)


def test_vit_parameter_counts_increase():
    counts = [sum(p.numel() for p in ModelRegistry.create(n).parameters()) for n in VIT_SIZES]
    assert counts == sorted(counts) and counts[0] > 5e5 and counts[-1] < 2e7


def test_norm_linear_groups_find_qkv_and_fc1():
    groups = norm_linear_groups(ModelRegistry.create("vit_t_cifar"))
    assert groups["blocks.0.ln1"] == ["blocks.0.attn.qkv"]
    assert groups["blocks.0.ln2"] == ["blocks.0.mlp.fc1"]
    assert "norm" not in groups  # the final norm feeds indexing, not a Linear


def _vit_with_outlier():
    torch.manual_seed(0)
    m = ModelRegistry.create("vit_t_cifar").eval()
    with torch.no_grad():
        for blk in m.blocks:
            blk.ln1.weight[3] = 60.0  # one systematic outlier channel feeding qkv
    return m


def test_smoothing_preserves_float_function():
    m = _vit_with_outlier()
    ref = copy.deepcopy(m)
    n = smooth(m, _loader(), SmoothQuantConfig(alpha=0.5), "cpu")
    assert n == 12  # ln1 and ln2 in 6 blocks
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(m(x), ref(x), rtol=1e-4, atol=1e-4)


def test_outlier_detection_and_smoothquant_reduces_w8a8_error():
    m = _vit_with_outlier()
    stats = outlier_stats(m, _loader())
    assert stats["blocks.0.attn.qkv"] > 10
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        ref = m(x)

    def w8a8(model):
        quantize_model(model, WeightSpec(bits=8), act_bits=8)
        calibrate_activations(model, _loader(), num_samples=32)
        with torch.no_grad():
            return float((model(x) - ref).norm() / ref.norm())

    plain = w8a8(copy.deepcopy(m))
    smoothed = copy.deepcopy(m)
    smooth(smoothed, _loader(), SmoothQuantConfig(alpha=0.5), "cpu")
    assert w8a8(smoothed) < plain


def test_smoothquant_method_in_pipeline():
    sec = QuantizationSection.from_dict(
        {
            "enabled": True,
            "method": "smoothquant",
            "calibration": {"num_samples": 16},
            "smoothquant": {"alpha": 0.6, "num_samples": 16},
        }
    )
    q = run_quantization(_vit_with_outlier(), sec, _loader(), "cpu")
    assert q(torch.randn(1, 3, 32, 32)).shape == (1, 10)
    with pytest.raises(ValueError, match="unknown smoothquant options"):
        SmoothQuantConfig.from_dict({"beta": 1})
