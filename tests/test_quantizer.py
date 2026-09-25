from __future__ import annotations

import numpy as np
import pytest
import torch

from edge_ai_compression.compression.quantization.quantizer import (
    ActObserver,
    WeightSpec,
    fake_quant,
    fake_quant_weight,
    integer_weight,
    qmax,
    weight_scale,
)
from edge_ai_compression.inference import packing

torch.manual_seed(0)


def test_qmax_and_bits_validation():
    assert qmax(8) == 127 and qmax(4) == 7
    with pytest.raises(ValueError):
        qmax(1)


def test_fake_quant_ste_gradient_and_clamp():
    x = torch.tensor([0.26, -3.0, 0.74], requires_grad=True)
    y = fake_quant(x, torch.tensor(0.5), bits=2)  # grid {-0.5, 0, 0.5}
    assert y.tolist() == [0.5, -0.5, 0.5]
    y.sum().backward()
    # STE passes gradient inside the range; clamped elements get none.
    assert x.grad.tolist() == [1.0, 0.0, 0.0]


def test_per_channel_rtn_matches_engine_int8_packing():
    w = torch.randn(16, 8, 3, 3)
    spec = WeightSpec(bits=8, granularity="per_channel")
    scale = weight_scale(w, spec)
    q = integer_weight(w, scale, spec).numpy()
    q_ref, s_ref = packing.quantize_rows_s8(w.reshape(16, -1).numpy())
    np.testing.assert_allclose(scale.reshape(-1).numpy(), s_ref, rtol=1e-6)
    assert np.array_equal(q, q_ref)


def test_per_group_matches_engine_w4_packing():
    w = torch.randn(6, 70)
    spec = WeightSpec(bits=4, granularity="per_group", group_size=32)
    scale = weight_scale(w, spec)
    fq = fake_quant_weight(w, scale, spec).numpy()
    q, s = packing.quantize_w4(w.numpy(), 32)
    np.testing.assert_allclose(fq, packing.dequantize_w4(q, s, 70, 32), rtol=1e-5, atol=1e-6)


def test_per_tensor_uses_single_scale():
    w = torch.randn(4, 5)
    scale = weight_scale(w, WeightSpec(granularity="per_tensor"))
    assert scale.numel() == 1 and float(scale) == pytest.approx(float(w.abs().max()) / 127)


def test_mse_weight_scale_reduces_error_for_heavy_tails():
    w = torch.randn(8, 512)
    w[:, 0] = 40.0  # outliers make min-max clipping wasteful at 4 bits
    base = WeightSpec(bits=4, granularity="per_channel")
    mse = WeightSpec(bits=4, granularity="per_channel", method="mse")
    err = {
        s.method: float((fake_quant_weight(w, weight_scale(w, s), s) - w).pow(2).sum())
        for s in (base, mse)
    }
    assert err["mse"] < err["minmax"]


def test_spec_validation():
    with pytest.raises(ValueError, match="group_size"):
        WeightSpec(granularity="per_group")
    with pytest.raises(ValueError, match="group_size"):
        WeightSpec(granularity="per_channel", group_size=32)


def test_observers():
    x = torch.cat([torch.randn(100_000), torch.tensor([50.0])])
    mm = ActObserver("minmax")
    pc = ActObserver("percentile", percentile=99.9)
    ms = ActObserver("mse", bits=8)
    for obs in (mm, pc, ms):
        for chunk in x.split(10_000):
            obs.observe(chunk)
    assert mm.scale() == pytest.approx(50.0 / 127)
    assert pc.scale() < mm.scale() / 5  # the single outlier is ignored
    assert ms.scale() <= mm.scale()
    assert ActObserver("minmax").scale() == 1.0  # nothing observed


def test_mse_per_group_picks_scales_per_group():
    w = torch.randn(2, 64)
    w[0, 5] = 30.0  # outlier only in the first group of row 0
    spec = WeightSpec(bits=4, granularity="per_group", group_size=32, method="mse")
    scale = weight_scale(w, spec)
    base = weight_scale(w, WeightSpec(bits=4, granularity="per_group", group_size=32))
    assert scale.shape == (2, 2, 1)
    assert float(scale[0, 1]) <= float(base[0, 1]) + 1e-9  # other groups chosen independently
