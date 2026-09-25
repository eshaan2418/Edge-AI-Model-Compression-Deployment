from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.compression.quantization.ptq import run_quantization
from edge_ai_compression.core.experiment import CompressionConfig, QuantizationSection
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.inference import kernels

NEEDS_KERNELS = pytest.mark.skipif(
    not kernels.available() and not kernels.kernels_required(), reason="C++ kernels not built"
)


def _resnet() -> nn.Module:
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar")
    m.train()
    with torch.no_grad():
        m(torch.randn(16, 3, 32, 32))
    return m.eval()


def _loader() -> DataLoader:
    torch.manual_seed(1)
    x = torch.randn(48, 3, 32, 32)
    return DataLoader(TensorDataset(x, torch.zeros(48, dtype=torch.long)), batch_size=16)


def _sec(**kw) -> QuantizationSection:
    base = {"enabled": True, "calibration": {"num_samples": 32}}
    return QuantizationSection.from_dict({**base, **kw})


def test_config_parsing_and_errors():
    sec = _sec(weight_bits=4, act_bits=None, group_size=32)
    assert sec.granularity == "per_group" and sec.tag == "rtn:w4afp:g32"
    assert _sec().tag == "rtn:w8a8:per_channel"
    with pytest.raises(ValueError, match="replaced by quantization.method"):
        QuantizationSection.from_dict({"mode": "dynamic_linear"})
    with pytest.raises(ValueError, match="unknown quantization method"):
        QuantizationSection.from_dict({"method": "gptq"})
    with pytest.raises(ValueError, match="unknown quantization keys"):
        QuantizationSection.from_dict({"bits": 8})
    assert CompressionConfig.from_dict({"quantization": {"enabled": True}}).quantization.enabled
    sec = QuantizationSection(enabled=True, method="rtn", options={"x": 1})
    assert sec.to_dict()["rtn"] == {"x": 1} and "options" not in sec.to_dict()


def test_activation_quant_requires_calibration_data():
    with pytest.raises(ValueError, match="calibration data"):
        run_quantization(_resnet(), _sec(), None, "cpu")


def _engine_matches_simulation(sec: QuantizationSection, layer_tol: float) -> None:
    """Lowering is exact per layer; end to end the two agree up to rounding-boundary flips.

    Given identical inputs, every lowered layer must reproduce its simulated
    layer to float precision. End to end, the engine (exact int32 accumulate,
    one rescale) and the simulation (float accumulate of dequantized products)
    differ at ~1e-7, which occasionally flips a value sitting exactly on a
    rounding boundary at the next layer's input quantization; those one-step
    differences compound through a deep net. So the end-to-end check is loose.
    """
    from edge_ai_compression.compression.quantization.modules import quant_layers
    from edge_ai_compression.inference.engine import compile_model

    m = run_quantization(_resnet(), sec, _loader(), "cpu")
    x = next(iter(_loader()))[0][:2]
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = [
        layer.register_forward_hook(
            lambda _m, a, out, n=name: captured.__setitem__(n, (a[0][0].clone(), out[0].clone()))
        )
        for name, layer in quant_layers(m)
    ]
    with torch.no_grad():
        sim = m(x).numpy()
    for h in hooks:
        h.remove()
    engine = compile_model(copy.deepcopy(m), "quant").prepare()
    gemms = {n.gemm.name: n.gemm for n in engine.nodes if n.gemm is not None}
    for name, (inp, ref) in captured.items():
        gemm = gemms[name.replace(".", "_")]
        out = gemm.run(inp.numpy(), None)
        ref = ref.numpy()
        if gemm.relu:
            ref = ref.clip(min=0)
        rel = float(((out - ref) ** 2).sum() ** 0.5 / ((ref**2).sum() ** 0.5 + 1e-12))
        assert rel < layer_tol, (name, rel)
    out = engine(x.numpy())
    rel = float(((out - sim) ** 2).sum() ** 0.5 / (sim**2).sum() ** 0.5)
    assert rel < 0.05, rel
    assert (out.argmax(1) == sim.argmax(1)).all()


@NEEDS_KERNELS
def test_engine_reproduces_w8a8_static_simulation():
    _engine_matches_simulation(_sec(), 1e-5)


@NEEDS_KERNELS
def test_engine_reproduces_w4_weight_only_group_simulation():
    _engine_matches_simulation(_sec(weight_bits=4, act_bits=None, group_size=32), 1e-5)


@NEEDS_KERNELS
def test_engine_reproduces_w4a8_on_int8_kernel():
    _engine_matches_simulation(_sec(weight_bits=4), 1e-5)


@NEEDS_KERNELS
def test_engine_mode_checks_and_unsupported_combo():
    from edge_ai_compression.inference.engine import compile_model

    with pytest.raises(ValueError, match="mode 'quant'"):
        compile_model(_resnet(), "quant")
    q = run_quantization(_resnet(), _sec(weight_bits=4, group_size=32), _loader(), "cpu")
    with pytest.raises(ValueError, match="mode 'quant'"):
        compile_model(q, "int8")
    with pytest.raises(NotImplementedError, match="has no kernel"):
        compile_model(q, "quant")  # per-group weights with int8 activations
