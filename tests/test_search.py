from __future__ import annotations

from edge_ai_compression.optimization.search.bayesian_search import BayesianSearch


def test_bayesian_search_runs():
    bounds = {"x": (0.0, 1.0)}

    def objective(cfg: dict) -> dict:
        return {
            "accuracy": float(cfg["x"]),
            "latency_ms_mean": 1.0,
            "size_mb": 1.0,
            "peak_ram_mib": 1.0,
        }

    def score_fn(r: dict) -> float:
        return r["accuracy"]

    s = BayesianSearch(bounds, random_state=1)
    out = s.run(5, objective, score_fn)
    assert len(out) == 5
