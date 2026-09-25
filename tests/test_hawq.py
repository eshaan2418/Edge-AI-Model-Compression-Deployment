from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.analysis.hessian import hutchinson_traces
from edge_ai_compression.compression.quantization.hawq import HAWQConfig, allocate_bits
from edge_ai_compression.compression.quantization.modules import quant_layers
from edge_ai_compression.compression.quantization.ptq import run_quantization
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.core.experiment import QuantizationSection
from edge_ai_compression.core.registry import ModelRegistry


def test_hutchinson_matches_exact_trace_of_quadratic():
    # loss = mean((X w)^2) -> Hessian = 2/N X^T X, trace = 2/N ||X||_F^2
    torch.manual_seed(0)
    x = torch.randn(64, 10)
    w = torch.randn(10, requires_grad=True)
    exact = 2.0 / 64 * float(x.pow(2).sum())
    est = hutchinson_traces(lambda: (x @ w).pow(2).mean(), {"w": w}, max_iters=2000, tol=1e-6)
    assert est["w"] == pytest.approx(exact, rel=0.05)


def test_hutchinson_blocks_are_separate():
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(5, requires_grad=True)
    est = hutchinson_traces(
        lambda: (a**2).sum() + 10 * (b**2).sum(), {"a": a, "b": b}, max_iters=20
    )
    assert est["a"] == pytest.approx(6.0) and est["b"] == pytest.approx(100.0)


def test_allocation_respects_budget_and_protects_sensitive_layers():
    torch.manual_seed(0)
    weights = {n: torch.randn(64, 64) for n in ("a", "b", "c", "d")}
    traces = {"a": 100.0, "b": 1e-3, "c": 1e-3, "d": 1e-3}
    bits = allocate_bits(weights, traces, HAWQConfig(avg_bits=5.0), WeightSpec(bits=8))
    assert bits["a"] == 8 and sum(bits.values()) / 4 <= 5.0
    pinned = allocate_bits(weights, traces, HAWQConfig(avg_bits=6.0), WeightSpec(), {"d": 8})
    assert pinned["d"] == 8
    with pytest.raises(ValueError, match="no bit allocation"):
        allocate_bits(weights, traces, HAWQConfig(avg_bits=3.0), WeightSpec())


def test_hawq_pipeline_assigns_mixed_bits_within_budget():
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar").eval()
    loader = DataLoader(
        TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))), batch_size=16
    )
    sec = QuantizationSection.from_dict(
        {
            "enabled": True,
            "method": "hawq",
            "calibration": {"num_samples": 16},
            "hawq": {"avg_bits": 6.0, "hutchinson_iters": 5, "hutchinson_samples": 16},
        }
    )
    assert sec.tag == "hawq:wavg6.0a8:per_channel:minmax"
    q = run_quantization(m, sec, loader, "cpu")
    layers = quant_layers(q)
    bits = {n: layer.spec.bits for n, layer in layers}
    assert bits["conv1"] == 8 and bits["fc"] == 8
    n = {name: layer.weight.numel() for name, layer in layers}
    assert sum(bits[k] * n[k] for k in n) / sum(n.values()) <= 6.0 + 1e-9
    assert set(bits.values()) == {4, 8}
