from __future__ import annotations

import numpy as np

from edge_ai_compression.optimization.search.nsga2 import nsga2_minimize


def test_nsga2_runs():
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def eval_fn(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2], dtype=np.float64)

    pop = nsga2_minimize(eval_fn, bounds, pop_size=12, generations=4, seed=1)
    assert len(pop) == 12
