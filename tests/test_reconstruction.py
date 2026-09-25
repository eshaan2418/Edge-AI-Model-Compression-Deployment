from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.compression.quantization.modules import (
    fold_bn,
    quant_layers,
    quantize_model,
)
from edge_ai_compression.compression.quantization.ptq import run_quantization
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.compression.quantization.reconstruction import (
    ReconstructionConfig,
    reconstruct,
    units,
)
from edge_ai_compression.core.experiment import QuantizationSection
from edge_ai_compression.core.registry import ModelRegistry


def _loader(n=64, shape=(3, 32, 32)):
    torch.manual_seed(1)
    return DataLoader(
        TensorDataset(torch.randn(n, *shape), torch.randint(0, 10, (n,))), batch_size=16
    )


def _resnet():
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar")
    m.train()
    with torch.no_grad():
        m(torch.randn(16, 3, 32, 32))
    return fold_bn(m.eval())


def test_units_layer_and_block_granularity():
    m = _resnet()
    quantize_model(m, WeightSpec(bits=4), act_bits=None)
    assert len(units(m, "layer")) == 21
    blocks = units(m, "block")
    assert blocks[0] == "conv1" and blocks[-1] == "fc" and "layer2.0" in blocks
    assert len(blocks) == 10  # stem + 8 BasicBlocks + fc
    with pytest.raises(ValueError):
        units(m, "network")


def _layer_error(q: nn.Module, fp: nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        return float((q(x) - fp(x)).pow(2).mean())


def test_adaround_beats_nearest_rounding_on_reconstruction_error():
    torch.manual_seed(0)
    fp = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1)).eval()
    loader = _loader(128, (8, 8, 8))
    x = torch.cat([b for b, _ in loader])
    rtn = copy.deepcopy(fp)
    quantize_model(rtn, WeightSpec(bits=3), act_bits=None, first_last_bits=None)
    ada = copy.deepcopy(rtn)
    stats = reconstruct(
        ada, fp, loader, ReconstructionConfig(iters=600, num_samples=128), granularity="layer"
    )
    assert set(stats) == {"0"}
    layer = dict(quant_layers(ada))["0"]
    assert layer.hard_round and layer.integer_weight().abs().max() <= 3
    assert _layer_error(ada, fp, x) < _layer_error(rtn, fp, x)


@pytest.mark.parametrize("method,extra", [("adaround", {}), ("brecq", {"fisher": True})])
def test_pipeline_methods_run_and_hard_round(method, extra):
    opts = {"iters": 5, "num_samples": 32, "batch_size": 8, **extra}
    sec = QuantizationSection.from_dict(
        {
            "enabled": True,
            "method": method,
            "weight_bits": 4,
            "calibration": {"num_samples": 32},
            method: opts,
        }
    )
    m = run_quantization(ModelRegistry.create("resnet18_cifar").eval(), sec, _loader(), "cpu")
    layers = quant_layers(m)
    assert all(layer.alpha is not None and layer.hard_round for _, layer in layers)
    assert all(not layer.alpha.requires_grad and layer.weight.requires_grad for _, layer in layers)


def test_unknown_reconstruction_option():
    with pytest.raises(ValueError, match="unknown reconstruction options"):
        ReconstructionConfig.from_dict({"iterations": 5})
