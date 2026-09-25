from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from edge_ai_compression.compression.quantization.modules import (
    QuantConv2d,
    QuantLinear,
    fold_bn,
    quant_layers,
    quantize_model,
)
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.inference import packing


def _resnet() -> nn.Module:
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar")
    m.train()
    with torch.no_grad():
        for _ in range(2):
            m(torch.randn(8, 3, 32, 32))
    return m.eval()


def test_fold_bn_preserves_outputs_and_class():
    m = _resnet()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = m(x)
        folded = fold_bn(copy.deepcopy(m))
        out = folded(x)
    assert type(folded) is type(m)
    assert not any(isinstance(mod, nn.BatchNorm2d) for mod in folded.modules())
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_quantize_model_first_last_policy_and_order():
    m = fold_bn(_resnet())
    order = quantize_model(m, WeightSpec(bits=4), act_bits=8)
    layers = dict(quant_layers(m))
    assert len(order) == 21 and order[0] == "conv1" and order[-1] == "fc"
    assert layers["conv1"].spec.bits == 8 and layers["fc"].spec.bits == 8
    assert layers["layer1.0.conv1"].spec.bits == 4
    assert isinstance(layers["conv1"], QuantConv2d) and isinstance(layers["fc"], QuantLinear)


def test_w8_per_channel_close_to_float_and_act_quant_off_until_calibrated():
    m = fold_bn(_resnet())
    ref_model = copy.deepcopy(m)
    quantize_model(m, WeightSpec(bits=8), act_bits=8)
    assert not any(layer.act_enabled for _, layer in quant_layers(m))
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        ref, out = ref_model(x), m(x)
    assert float((out - ref).norm() / ref.norm()) < 0.02


def test_integer_weight_matches_engine_packing():
    conv = nn.Conv2d(3, 8, 3)
    q = QuantConv2d(conv, WeightSpec(bits=8), act_bits=None)
    codes = q.integer_weight().numpy()
    ref, _ = packing.quantize_rows_s8(conv.weight.detach().reshape(8, -1).numpy())
    assert np.array_equal(codes, ref)
    # quant_weight is exactly codes * scale
    torch.testing.assert_close(
        q.quant_weight().reshape(8, -1), q.integer_weight().float() * q.w_scale.reshape(-1, 1)
    )


def test_adaround_init_reproduces_float_weight_then_hard_rounds_to_grid():
    lin = nn.Linear(16, 4)
    q = QuantLinear(lin, WeightSpec(bits=4), act_bits=None)
    q.init_adaround()
    w_soft = q.quant_weight()
    # Soft rounding at init equals floor + frac, i.e. the float weight (inside the clamp range).
    torch.testing.assert_close(w_soft, lin.weight, rtol=1e-3, atol=1e-3)
    q.hard_round = True
    codes = q.integer_weight().float()
    torch.testing.assert_close(q.quant_weight().reshape(4, -1), codes * q.w_scale.reshape(-1, 1))
    assert codes.abs().max() <= 7


def test_act_quant_applies_once_enabled():
    lin = QuantLinear(nn.Linear(4, 2), WeightSpec(bits=8), act_bits=8)
    x = torch.tensor([[0.1, 0.2, 0.3, 10.0]])
    lin.a_scale.fill_(0.1)
    lin.act_enabled = True
    torch.testing.assert_close(lin.quant_input(x), torch.tensor([[0.1, 0.2, 0.3, 10.0]]))
    lin.a_scale.fill_(0.05)  # 10.0 now clips to 127 * 0.05 = 6.35
    assert float(lin.quant_input(x)[0, 3]) == float(np.float32(127 * 0.05))
