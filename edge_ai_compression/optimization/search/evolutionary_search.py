from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np


class EvolutionarySearch:
    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        population_size: int = 12,
        generations: int = 8,
        mutation_scale: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.bounds = bounds
        self.population_size = population_size
        self.generations = generations
        self.mutation_scale = mutation_scale
        self.keys = list(bounds.keys())
        self.dim = len(self.keys)
        random.seed(seed)
        self._rng = np.random.default_rng(seed)

    def _random_individual(self) -> np.ndarray:
        return self._rng.random(self.dim)

    def _decode(self, vec: np.ndarray) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for i, k in enumerate(self.keys):
            lo, hi = self.bounds[k]
            cfg[k] = float(lo + vec[i] * (hi - lo))
        return cfg

    def _fitness(self, vec: np.ndarray, fitness_fn: Callable[[dict[str, Any]], float]) -> float:
        return fitness_fn(self._decode(vec))

    def evolve(
        self,
        fitness_fn: Callable[[dict[str, Any]], float],
    ) -> list[dict[str, Any]]:
        population = [self._random_individual() for _ in range(self.population_size)]
        history: list[dict[str, Any]] = []

        for gen in range(self.generations):
            scored = [(ind, self._fitness(ind, fitness_fn)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            parents = [ind for ind, _ in scored[: max(2, self.population_size // 2)]]

            offspring: list[np.ndarray] = []
            while len(offspring) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                alpha = random.random()
                child = alpha * p1 + (1 - alpha) * p2
                noise = self._rng.normal(0, self.mutation_scale, size=self.dim)
                child = np.clip(child + noise, 0, 1)
                offspring.append(child)
            population = offspring
            best_ind, best_score = max(scored, key=lambda x: x[1])
            history.append(
                {
                    "generation": gen,
                    "best_score": best_score,
                    "best_config": self._decode(best_ind),
                }
            )
        return history
