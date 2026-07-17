from __future__ import annotations

import math

import pytest

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.hardware.profiles import HardwareProfile, get_profile
from edge_ai_compression.theory.model_complexity import (
    estimate_activation_bytes,
    weight_bytes,
)
from edge_ai_compression.theory.roofline import (
    analyze,
    arithmetic_intensity,
    main,
    roofline,
    theoretical_min_latency_ms,
)


def test_arithmetic_intensity():
    assert arithmetic_intensity(2e9, 2e8) == 10.0
    with pytest.raises(ValueError):
        arithmetic_intensity(1.0, 0.0)


def test_roofline_compute_bound():
    # intensity 10 >= ridge 5 -> compute-bound; attainable = min(100, 20*10) = 100.
    rl = roofline(10.0, peak_gflops=100.0, mem_bandwidth_gbs=20.0)
    assert rl["ridge_intensity"] == 5.0
    assert rl["bound"] == "compute"
    assert rl["attainable_gflops"] == 100.0


def test_roofline_memory_bound():
    # intensity 2 < ridge 5 -> memory-bound; attainable = min(100, 20*2) = 40.
    rl = roofline(2.0, peak_gflops=100.0, mem_bandwidth_gbs=20.0)
    assert rl["bound"] == "memory"
    assert rl["attainable_gflops"] == 40.0


def test_theoretical_min_latency():
    # FLOPs 2e9 at 100 GFLOP/s = 20 ms; bytes 2e8 at 20 GB/s = 10 ms; floor = 20.
    b = theoretical_min_latency_ms(2e9, 2e8, peak_gflops=100.0, mem_bandwidth_gbs=20.0)
    assert math.isclose(b["compute_bound_ms"], 20.0)
    assert math.isclose(b["memory_bound_ms"], 10.0)
    assert b["lower_bound_ms"] == 20.0


def test_activation_and_weight_bytes_positive():
    model = ModelRegistry.create("small_cnn_student", num_classes=10)
    assert weight_bytes(model) > 0
    assert estimate_activation_bytes(model, (1, 3, 32, 32)) > 0
    # int8 weights are 1/4 the FP32 size.
    assert weight_bytes(model, dtype_bytes=1) == weight_bytes(model, dtype_bytes=4) / 4


def test_analyze_reports_efficiency_below_one():
    model = ModelRegistry.create("small_cnn_student", num_classes=10)
    profile = get_profile("raspberry_pi")
    # A deliberately huge measured latency -> efficiency well under 1.
    report = analyze(model, (1, 3, 32, 32), profile, measured_latency_ms=10_000.0)
    assert report["flops"] > 0
    assert report["roofline"]["bound"] in {"compute", "memory"}
    assert 0.0 < report["roofline_efficiency"] < 1.0
    # Lower bound is a floor: it must not exceed the measured latency here.
    assert report["latency_bounds_ms"]["lower_bound_ms"] <= 10_000.0


def test_analyze_requires_ceilings():
    model = ModelRegistry.create("small_cnn_student", num_classes=10)
    bare = HardwareProfile(
        name="bare",
        max_latency_ms=100.0,
        max_size_mb=100.0,
        max_ram_mb=100.0,
        preferred_export_format="onnx",
        notes="no ceilings",
    )
    with pytest.raises(ValueError):
        analyze(model, (1, 3, 32, 32), bare)


def test_cli_smoke(tmp_path):
    out = tmp_path / "roofline.json"
    rc = main(["--model", "small_cnn_student", "--profile", "cpu", "--out", str(out)])
    assert rc == 0
    assert out.is_file()


def test_cli_unknown_profile_returns_1():
    assert main(["--model", "small_cnn_student", "--profile", "nope"]) == 1
