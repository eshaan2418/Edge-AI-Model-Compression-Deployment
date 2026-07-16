from __future__ import annotations

import pytest

from edge_ai_compression.hardware.profiles import available_profiles, get_profile
from edge_ai_compression.hardware.scoring import score_candidate


def test_feasible_candidate_on_cpu():
    metrics = {"accuracy": 0.9, "latency_ms": 10.0, "size_mb": 5.0, "ram_mb": 100.0}
    res = score_candidate(metrics, "cpu")
    assert res.feasible is True
    assert res.violations == []
    assert res.score > 0


def test_infeasible_on_microcontroller():
    # A ResNet-scale model cannot fit an MCU: size and RAM blow the budget.
    metrics = {"accuracy": 0.9, "latency_ms": 10.0, "size_mb": 42.0, "ram_mb": 300.0}
    res = score_candidate(metrics, "microcontroller_sim")
    assert res.feasible is False
    assert len(res.violations) >= 2
    # Infeasible must score below an otherwise-identical feasible candidate on cpu.
    feasible = score_candidate(metrics, "cpu")
    assert res.score < feasible.score


def test_metric_aliases_accepted():
    # benchmark_model uses latency_ms_mean / peak_ram_mib naming.
    metrics = {"accuracy": 0.8, "latency_ms_mean": 20.0, "size_mb": 10.0, "peak_ram_mib": 200.0}
    res = score_candidate(metrics, "raspberry_pi")
    assert res.feasible is True
    assert "latency_ms" in res.utilization
    assert "ram_mb" in res.utilization


def test_missing_metrics_do_not_crash():
    res = score_candidate({"accuracy": 0.7}, "smartphone")
    # No resource metrics -> no violations possible, but flagged as unknown.
    assert res.violations == []
    assert "unknown_metrics" in res.explanation


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        get_profile("quantum_toaster")


def test_all_profiles_present():
    assert set(available_profiles()) == {
        "cpu",
        "raspberry_pi",
        "smartphone",
        "microcontroller_sim",
    }
