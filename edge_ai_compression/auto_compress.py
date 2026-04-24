from __future__ import annotations

import argparse
import copy
from dataclasses import replace

import torch

from edge_ai_compression.core.experiment import CompressionConfig, ExperimentConfig
from edge_ai_compression.core.runner import ExperimentRunner
from edge_ai_compression.optimization.pareto.frontier import ParetoOptimizer
from edge_ai_compression.optimization.scoring.cost_functions import scalarized_objective
from edge_ai_compression.optimization.search.random_search import RandomSearch


class AutoCompressor:
    def __init__(
        self,
        *,
        max_latency_ms: float,
        max_size_mb: float,
        min_accuracy: float,
        max_ram_mib: float | None = None,
    ) -> None:
        self.max_latency_ms = max_latency_ms
        self.max_size_mb = max_size_mb
        self.min_accuracy = min_accuracy
        self.max_ram_mib = max_ram_mib

    def _valid(self, r: dict) -> bool:
        if r["accuracy"] < self.min_accuracy:
            return False
        if r["latency_ms_mean"] > self.max_latency_ms:
            return False
        if r["size_mb"] > self.max_size_mb:
            return False
        if self.max_ram_mib is not None and r["peak_ram_mib"] > self.max_ram_mib:
            return False
        return True

    def score(self, r: dict) -> float:
        return scalarized_objective(r)

    def find_best(self, results: list[dict]) -> dict | None:
        valid = [r for r in results if self._valid(r)]
        if not valid:
            return None
        return max(valid, key=self.score)


def _torch_device(device: str) -> str:
    if device in ("raspberry_pi", "mobile", "cpu"):
        return "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _base_config(device: str) -> ExperimentConfig:
    return ExperimentConfig(
        model="resnet18_cifar",
        device=_torch_device(device),
        compression=CompressionConfig(),
        benchmark={"latency_repeats": 40, "latency_warmup": 2},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-aware auto compression search")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--max-latency-ms", type=float, default=50.0)
    parser.add_argument("--max-size-mb", type=float, default=50.0)
    parser.add_argument("--min-accuracy", type=float, default=0.5)
    parser.add_argument("--max-ram-mib", type=float, default=None)
    parser.add_argument("--budget", type=int, default=4, help="Random search trials")
    args = parser.parse_args()

    space = {
        "prune_amount": (0.1, 0.7),
    }
    search = RandomSearch(space, seed=0)

    auto = AutoCompressor(
        max_latency_ms=args.max_latency_ms,
        max_size_mb=args.max_size_mb,
        min_accuracy=args.min_accuracy,
        max_ram_mib=args.max_ram_mib,
    )

    def objective(sample: dict) -> dict:
        cfg = _base_config(args.device)
        comp = copy.deepcopy(cfg.compression)
        comp.pruning.enabled = True
        comp.pruning.amount = float(sample["prune_amount"])
        cfg = replace(cfg, compression=comp)
        res = ExperimentRunner(cfg).run()
        d = res.to_dict()
        d["prune_amount"] = sample["prune_amount"]
        return d

    results = [objective(search.sample()) for _ in range(args.budget)]
    best = auto.find_best(results)
    pareto = ParetoOptimizer().compute_frontier(results)

    print("--- Auto-compress summary ---")
    for i, r in enumerate(results):
        print(f"trial {i}: acc={r['accuracy']:.3f} lat={r['latency_ms_mean']:.2f}ms size={r['size_mb']:.2f}MB")
    print(f"Pareto frontier size: {len(pareto)}")
    if best:
        print(f"Best feasible by scalarized score: {best}")
    else:
        print("No configuration satisfied all constraints; relax limits or increase budget.")


if __name__ == "__main__":
    main()
