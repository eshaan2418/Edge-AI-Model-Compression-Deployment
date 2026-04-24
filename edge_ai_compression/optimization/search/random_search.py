from __future__ import annotations

import random
from typing import Any, Callable


class RandomSearch:
    def __init__(self, space: dict[str, tuple[Any, Any]], seed: int | None = None) -> None:
        """space maps name -> (low, high) for numeric uniform sampling."""
        self.space = space
        if seed is not None:
            random.seed(seed)

    def sample(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for k, (lo, hi) in self.space.items():
            if isinstance(lo, int) and isinstance(hi, int):
                cfg[k] = random.randint(lo, hi)
            else:
                cfg[k] = random.uniform(float(lo), float(hi))
        return cfg

    def run(self, budget: int, objective: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
        return [objective(self.sample()) for _ in range(budget)]
