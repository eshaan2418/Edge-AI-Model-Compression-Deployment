from __future__ import annotations

import json

from edge_ai_compression.benchmarking.benchmark_model import benchmark_model
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.theory.model_complexity import model_summary


def test_benchmark_model_report():
    model = ModelRegistry.create("small_cnn_student")
    report = benchmark_model(model, (1, 3, 32, 32), warmup=1, repeats=3, measure_memory=False)
    lat = report["latency_ms"]
    assert lat["mean"] > 0
    # Percentiles are monotonic non-decreasing.
    assert lat["p50"] <= lat["p95"] <= lat["p99"]
    assert report["num_parameters"] > 0
    assert report["size_mb"] > 0
    assert report["throughput_ips"] > 0
    # Must be JSON-serializable end to end.
    json.dumps(report)


def test_model_summary_fields():
    model = ModelRegistry.create("small_cnn_student")
    s = model_summary(model)
    assert s["num_parameters"] >= s["num_trainable_parameters"] > 0
    assert 0.0 <= s["weight_sparsity"] <= 1.0
    assert s["num_modules"] > 0
    assert isinstance(s["layer_type_histogram"], dict)
