from __future__ import annotations

import pandas as pd
import pytest

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.inference import kernels
from edge_ai_compression.inference.kernel_bench import GEMM_OPS, KernelSpec
from edge_ai_compression.inference.roofline import (
    MachinePeaks,
    peaks_from_rows,
    roofline_point,
    roofline_table,
)
from edge_ai_compression.utils.config_loader import load_yaml

SMOKE = "edge_ai_compression/configs/studies/smoke_kernel_study.yml"


def test_spec_validation():
    with pytest.raises(ValueError, match="unknown"):
        KernelSpec("gemm_fp8", 1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        KernelSpec("gemm_f32", 0, 1, 1)
    with pytest.raises(ValueError, match="divisible by 4"):
        KernelSpec("gemm_sparse24", 1, 1, 6)
    assert KernelSpec("triad").is_peak


def test_roofline_point_math():
    peaks = MachinePeaks(fp32_gflops=100.0, s8_gops=400.0, bandwidth_gbps=50.0)
    assert peaks.ridge("gemm_f32") == 2.0 and peaks.ridge("gemm_s8") == 8.0
    mem = roofline_point("gemm_f32", ai=1.0, achieved=25.0, peaks=peaks)
    assert mem == {"attainable": 50.0, "efficiency": 0.5, "regime": "memory"}
    comp = roofline_point("gemm_s8", ai=10.0, achieved=200.0, peaks=peaks)
    assert comp["regime"] == "compute" and comp["attainable"] == 400.0


def test_peaks_from_rows_requires_single_machine():
    rows = pd.DataFrame(
        {
            "op": ["peak_fma_f32", "peak_dot_s8", "triad"],
            "achieved": [1.0, 2.0, 3.0],
            "fingerprint_hash": ["a", "a", "b"],
        }
    )
    with pytest.raises(ValueError, match="machines"):
        peaks_from_rows(rows)


@pytest.mark.skipif(not kernels.available() and not kernels.kernels_required(), reason="no kernels")
def test_smoke_study_logs_every_op_and_supports_roofline(tmp_path):
    from edge_ai_compression.experiment_db.paths import kernel_benchmarks_csv
    from edge_ai_compression.experiments.run_kernel_study import build_specs, run_study

    cfg = load_yaml(SMOKE)
    cfg["results_dir"] = str(tmp_path)
    specs = build_specs(cfg)
    run_study(cfg)
    df = pd.read_csv(kernel_benchmarks_csv(tmp_path))
    assert len(df) == len(specs)
    assert set(df["op"]) >= set(GEMM_OPS) | {"peak_fma_f32", "peak_dot_s8", "triad"}
    gemm = df[df["op"].str.startswith("gemm_")]
    assert (gemm["latency_median_us"] > 0).all() and (gemm["achieved"] > 0).all()
    assert (gemm["latency_median_ci_lo_us"] <= gemm["latency_median_us"]).all()
    table = roofline_table(df, peaks_from_rows(df))
    assert set(table["regime"]) <= {"memory", "compute"}
    csr = df[df["op"] == "gemm_csr"].sort_values("sparsity")
    assert csr["flops"].iloc[0] > csr["flops"].iloc[-1]  # useful FLOPs shrink with sparsity


@pytest.mark.skipif(not kernels.available() and not kernels.kernels_required(), reason="no kernels")
def test_model_gemm_shapes_resnet18():
    from edge_ai_compression.experiments.run_kernel_study import model_gemm_shapes

    shapes = model_gemm_shapes("resnet18_cifar", [3, 32, 32])
    assert (64, 1024, 27) in shapes and (10, 1, 512) in shapes


def test_bench_config_reused():
    assert BenchmarkConfig.from_dict(load_yaml(SMOKE)["benchmark"]).process_repeats == 2
