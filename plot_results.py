#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-id", default=None)
    p.add_argument("--results", type=Path, default=Path("results/experiments.csv"))
    p.add_argument("--out", type=Path, default=Path("results/plots"))
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Install optional [viz]: pip install 'edge-ai-compression[viz]'")
        return

    if not args.results.is_file():
        print(f"No {args.results}")
        return
    df = pd.read_csv(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.scatter(df["latency_mean"], df["accuracy"], alpha=0.7)
    plt.xlabel("Latency mean (ms)")
    plt.ylabel("Accuracy")
    p = args.out / "pareto_accuracy_latency.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
