from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.benchmarking.config import BACKENDS, BenchmarkConfig
from edge_ai_compression.benchmarking.evaluator import Evaluator
from edge_ai_compression.benchmarking.isolation import benchmark_model
from edge_ai_compression.inference import kernels
from edge_ai_compression.inference.backends import get_backend, run_batched

NEEDS_KERNELS = pytest.mark.skipif(
    not kernels.available() and not kernels.kernels_required(), reason="C++ kernels not built"
)
TINY = BenchmarkConfig(warmup_iters=1, min_warmup_s=0.0, iters=3, process_repeats=1)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    ).eval()


def _needs(name: str):
    return [NEEDS_KERNELS] if name.startswith("edge_") else []


@pytest.mark.parametrize(
    "name", [pytest.param(b, marks=_needs(b)) for b in BACKENDS if b != "edge_quant"]
)
def test_every_backend_exports_loads_and_runs_in_worker(name, tmp_path):
    m = _model()
    x = torch.randn(1, 3, 8, 8)
    backend = get_backend(name)
    path = backend.export(m, (1, 3, 8, 8), tmp_path)
    out = backend.load(path, 1)(x.numpy())
    with torch.no_grad():
        ref = m(x).numpy()
    tol = 0.15 if name in ("edge_int8", "edge_w4", "edge_sparse24") else 1e-4
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol)
    cfg = BenchmarkConfig(**{**TINY.to_dict(), "backend": name})
    run = benchmark_model(m, (1, 3, 8, 8), cfg)[0]
    assert run.trace_ns.shape == (3,)


def test_run_batched_pads_tail():
    seen = []

    def run(x):
        seen.append(x.shape[0])
        return x.reshape(len(x), -1)[:, :2]

    out = run_batched(run, np.arange(5 * 4, dtype=np.float32).reshape(5, 4), batch=2)
    assert seen == [2, 2, 2] and out.shape == (5, 2)


def test_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        BenchmarkConfig(backend="tflite")
    with pytest.raises(ValueError, match="backend"):
        get_backend("tflite")


@NEEDS_KERNELS
def test_evaluator_uses_backend_for_accuracy_and_size():
    m = _model()
    loader = DataLoader(
        TensorDataset(torch.randn(6, 3, 8, 8), torch.randint(0, 4, (6,))), batch_size=4
    )
    reports = {}
    for name in ("torch_eager", "edge_f32", "edge_int8"):
        cfg = BenchmarkConfig(**{**TINY.to_dict(), "backend": name, "strict_environment": False})
        reports[name] = Evaluator("cpu", cfg).evaluate(m, loader)
    assert reports["edge_f32"].accuracy == reports["torch_eager"].accuracy
    assert reports["edge_int8"].size_mb < reports["edge_f32"].size_mb
    assert reports["edge_int8"].config.backend == "edge_int8"


def test_onnx_artifact_is_self_contained(tmp_path):
    m = _model()
    path = get_backend("onnxruntime").export(m, (1, 3, 8, 8), tmp_path)
    assert [f.name for f in tmp_path.iterdir()] == ["model.onnx"]
    weights = sum(p.numel() for p in m.parameters()) * 4
    assert path.stat().st_size > weights
