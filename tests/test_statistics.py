from __future__ import annotations

import numpy as np

from edge_ai_compression.benchmarking.ab_compare import main
from edge_ai_compression.benchmarking.statistics import (
    bootstrap_ci,
    cliffs_delta,
    coefficient_of_variation,
    compare_latency,
    mann_whitney_u,
    outlier_fraction,
)


def _samples(mean, std, n, seed):
    return np.random.default_rng(seed).normal(mean, std, n).tolist()


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    xs = _samples(10.0, 1.0, 400, seed=1)
    ci = bootstrap_ci(xs, statistic="median", confidence=0.95, seed=7)
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert abs(ci["point"] - 10.0) < 0.3
    # Same seed -> identical result.
    assert bootstrap_ci(xs, seed=7) == ci


def test_higher_confidence_widens_interval():
    xs = _samples(10.0, 2.0, 300, seed=2)
    narrow = bootstrap_ci(xs, confidence=0.80, seed=3)
    wide = bootstrap_ci(xs, confidence=0.99, seed=3)
    assert (wide["high"] - wide["low"]) >= (narrow["high"] - narrow["low"])


def test_coefficient_of_variation_and_outliers():
    xs = [10.0] * 50
    assert coefficient_of_variation(xs) == 0.0
    assert outlier_fraction(xs) == 0.0
    with_outlier = [10.0] * 50 + [1000.0]
    assert outlier_fraction(with_outlier) > 0.0


def test_mann_whitney_detects_separated_distributions():
    a = _samples(10.0, 1.0, 200, seed=10)
    b = _samples(8.0, 1.0, 200, seed=11)
    mw = mann_whitney_u(a, b)
    assert mw["p_value"] < 0.01


def test_mann_whitney_identical_not_significant():
    a = _samples(10.0, 1.0, 200, seed=20)
    b = _samples(10.0, 1.0, 200, seed=21)
    mw = mann_whitney_u(a, b)
    assert mw["p_value"] > 0.05


def test_cliffs_delta_sign_and_bounds():
    a = _samples(10.0, 1.0, 200, seed=30)
    b = _samples(8.0, 1.0, 200, seed=31)
    d_ab = cliffs_delta(a, b)
    d_ba = cliffs_delta(b, a)
    assert -1.0 <= d_ab["delta"] <= 1.0
    assert d_ab["delta"] > 0  # a tends to be larger (slower)
    assert np.isclose(d_ab["delta"], -d_ba["delta"], atol=1e-9)
    assert d_ab["magnitude"] in {"small", "medium", "large"}


def test_compare_latency_reports_significant_speedup():
    slow = _samples(10.0, 0.5, 300, seed=40)
    fast = _samples(8.0, 0.5, 300, seed=41)
    rep = compare_latency(slow, fast, label_a="slow", label_b="fast", seed=1)
    # median(slow)/median(fast) ~ 1.25 -> fast is faster.
    assert rep["speedup_b_over_a"]["point"] > 1.15
    assert rep["significant"] is True
    assert "fast is significantly faster" in rep["verdict"]


def test_compare_latency_no_difference():
    a = _samples(10.0, 1.0, 300, seed=50)
    b = _samples(10.0, 1.0, 300, seed=51)
    rep = compare_latency(a, b, seed=2)
    assert rep["significant"] is False
    assert "No significant" in rep["verdict"]


def test_cli_smoke(tmp_path):
    out = tmp_path / "ab.json"
    rc = main(
        [
            "--model-a",
            "small_cnn_student",
            "--model-b",
            "small_cnn_student",
            "--repeats",
            "8",
            "--warmup",
            "1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.is_file()


def test_cli_unknown_model_returns_1():
    assert main(["--model-a", "nope", "--model-b", "small_cnn_student", "--repeats", "4"]) == 1
