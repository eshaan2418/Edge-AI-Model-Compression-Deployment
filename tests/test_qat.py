from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.compression.quantization.modules import QuantLinear, quant_layers
from edge_ai_compression.compression.quantization.ptq import run_quantization
from edge_ai_compression.compression.quantization.qat import QATConfig
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.core.experiment import QuantizationSection


def _task():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = (x[:, 0] > 0).long()  # linearly separable
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def test_lsq_scales_become_parameters_and_get_gradients():
    lin = QuantLinear(nn.Linear(16, 4), WeightSpec(bits=4), act_bits=8)
    lin.act_enabled = True
    lin.enable_lsq()
    assert isinstance(lin.w_scale, nn.Parameter) and isinstance(lin.a_scale, nn.Parameter)
    lin(torch.randn(8, 16)).sum().backward()
    assert lin.w_scale.grad is not None and lin.a_scale.grad is not None
    assert "w_scale" in dict(lin.named_parameters()) and "w_scale" in lin.state_dict()
    lin2 = QuantLinear(nn.Linear(4, 2), WeightSpec(bits=4), act_bits=None)
    lin2.init_adaround()
    with pytest.raises(ValueError, match="mutually exclusive"):
        lin2.enable_lsq()


def test_qat_pipeline_learns_and_updates_scales():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2)).eval()
    loader = _task()
    common = {
        "enabled": True,
        "weight_bits": 4,
        "first_last_bits": None,
        "calibration": {"num_samples": 64},
    }
    rtn = run_quantization(
        copy.deepcopy(base), QuantizationSection.from_dict(common), loader, "cpu"
    )
    sec = QuantizationSection.from_dict(
        {**common, "method": "qat", "qat": {"epochs": 30, "lr": 0.05}}
    )
    qat = run_quantization(copy.deepcopy(base), sec, loader, "cpu")
    x, y = next(iter(DataLoader(loader.dataset, batch_size=64)))
    with torch.no_grad():
        acc = {
            name: float((m(x).argmax(1) == y).float().mean())
            for name, m in (("rtn", rtn), ("qat", qat))
        }
    assert acc["qat"] > 0.9 and acc["qat"] >= acc["rtn"]
    rtn_scales = {n: layer.w_scale for n, layer in quant_layers(rtn)}
    assert any(
        not torch.equal(rtn_scales[n], layer.w_scale.detach()) for n, layer in quant_layers(qat)
    )


def test_unknown_qat_option():
    with pytest.raises(ValueError, match="unknown qat options"):
        QATConfig.from_dict({"epoch": 1})
