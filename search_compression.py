#!/usr/bin/env python3
"""Multi-objective compression search / Pareto reporting (Phase 6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from edge_ai_compression.optimization.pareto.frontier import ParetoOptimizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", default="pareto", choices=["pareto", "scalar"])
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "latency_mean", "size_mb", "ram_mb"],
    )
    p.add_argument("--results", type=Path, default=Path("results/experiments.csv"))
    p.add_argument("--out", type=Path, default=Path("results/pareto_frontier.json"))
    args = p.parse_args()

    if not args.results.is_file():
        print(f"No results at {args.results}; run experiments with experiment_db enabled first.")
        return

    df = pd.read_csv(args.results)
    rows = df.to_dict(orient="records")
    if args.objective == "pareto":
        maximize = tuple(m for m in args.metrics if m == "accuracy")
        minimize = tuple(m for m in args.metrics if m != "accuracy")
        front = ParetoOptimizer(maximize=maximize, minimize=minimize).compute_frontier(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(front, indent=2, default=str), encoding="utf-8")
        print(f"Pareto frontier: {len(front)} points → {args.out}")


if __name__ == "__main__":
    main()
