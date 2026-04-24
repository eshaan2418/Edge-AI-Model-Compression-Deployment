from __future__ import annotations

from typing import Any

from edge_ai_compression.optimization.pareto.dominance import dominates


class ParetoOptimizer:
    def __init__(
        self,
        maximize: tuple[str, ...] = ("accuracy",),
        minimize: tuple[str, ...] = ("latency_ms_mean", "size_mb", "peak_ram_mib"),
    ) -> None:
        self.maximize = maximize
        self.minimize = minimize

    def _dominates(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        return dominates(a, b, self.maximize, self.minimize)

    def compute_frontier(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frontier: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            dominated = False
            for j, o in enumerate(results):
                if i == j:
                    continue
                if self._dominates(o, r):
                    dominated = True
                    break
            if not dominated:
                frontier.append(r)
        return frontier
