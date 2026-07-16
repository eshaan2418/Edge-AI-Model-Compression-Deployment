from __future__ import annotations

import argparse

from edge_ai_compression.optimization.scoring.cost_functions import scalarized_objective
from edge_ai_compression.optimization.search.bayesian_search import BayesianSearch


def main() -> None:
    p = argparse.ArgumentParser(description="Toy Bayesian search on a synthetic objective")
    p.add_argument("--budget", type=int, default=12)
    args = p.parse_args()

    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}

    def objective(cfg: dict) -> dict:
        x, y = float(cfg["x"]), float(cfg["y"])
        acc = 1.0 / (1.0 + x * x + y * y)
        lat = 10.0 + (x * x + y * y) * 3.0
        size = 5.0 + abs(x) + abs(y)
        return {"accuracy": acc, "latency_ms_mean": lat, "size_mb": size, "peak_ram_mib": 128.0}

    search = BayesianSearch(bounds, random_state=0)

    def score_fn(r: dict) -> float:
        return scalarized_objective(r)

    out = search.run(args.budget, objective, score_fn)
    best = max(out, key=lambda z: z["score"])
    print("best:", best)


if __name__ == "__main__":
    main()
