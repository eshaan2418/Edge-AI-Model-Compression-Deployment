from __future__ import annotations

from edge_ai_compression.analysis.ablation import run_ablation


def main() -> None:
    arms = [
        ("noop", lambda: {"accuracy": 0.9, "latency_ms_mean": 5.0}),
        ("heavy_prune", lambda: {"accuracy": 0.7, "latency_ms_mean": 2.0}),
    ]
    print(run_ablation(arms))


if __name__ == "__main__":
    main()
