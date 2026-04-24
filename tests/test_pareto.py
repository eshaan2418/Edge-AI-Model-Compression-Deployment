from __future__ import annotations

from edge_ai_compression.optimization.pareto.frontier import ParetoOptimizer


def test_pareto_frontier():
    results = [
        {"accuracy": 0.9, "latency_ms_mean": 10.0, "size_mb": 5.0, "peak_ram_mib": 100.0},
        {"accuracy": 0.85, "latency_ms_mean": 5.0, "size_mb": 4.0, "peak_ram_mib": 90.0},
        {"accuracy": 0.5, "latency_ms_mean": 50.0, "size_mb": 20.0, "peak_ram_mib": 200.0},
    ]
    front = ParetoOptimizer().compute_frontier(results)
    assert len(front) >= 1
