from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianSearch:
    """Sequential model-based search with GP surrogate and UCB-style acquisition."""

    def __init__(self, bounds: dict[str, tuple[float, float]], random_state: int = 0) -> None:
        self.bounds = bounds
        self.dim = len(bounds)
        self.keys = list(bounds.keys())
        self.random_state = random_state
        self._xs: list[np.ndarray] = []
        self._ys: list[float] = []

    def _encode(self, cfg: dict[str, Any]) -> np.ndarray:
        vec = []
        for k in self.keys:
            lo, hi = self.bounds[k]
            v = float(cfg[k])
            vec.append((v - lo) / (hi - lo + 1e-9))
        return np.asarray(vec, dtype=np.float64)

    def _decode(self, vec: np.ndarray) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for i, k in enumerate(self.keys):
            lo, hi = self.bounds[k]
            cfg[k] = float(lo + vec[i] * (hi - lo))
        return cfg

    def sample_random(self) -> dict[str, Any]:
        rng = np.random.default_rng(self.random_state + len(self._xs))
        return self._decode(rng.random(self.dim))

    def sample_next(self) -> dict[str, Any]:
        if len(self._xs) < 2:
            return self.sample_random()
        X = np.stack(self._xs, axis=0)
        y = np.asarray(self._ys, dtype=np.float64)
        kernel = Matern(nu=2.5, length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=0)
        gp.fit(X, y)
        rng = np.random.default_rng(self.random_state + 17)
        best_score = -np.inf
        best_cfg: dict[str, Any] | None = None
        for _ in range(256):
            cand = rng.random(self.dim)
            mu, sigma = gp.predict(cand.reshape(1, -1), return_std=True)
            ucb = float(mu[0]) + 0.2 * float(sigma[0])
            if ucb > best_score:
                best_score = ucb
                best_cfg = self._decode(cand)
        assert best_cfg is not None
        return best_cfg

    def update_model(self, cfg: dict[str, Any], result: dict[str, Any], score_key: str = "score") -> None:
        self._xs.append(self._encode(cfg))
        self._ys.append(float(result[score_key]))

    def run(
        self,
        budget: int,
        objective: Callable[[dict[str, Any]], dict[str, Any]],
        score_fn: Callable[[dict[str, Any]], float],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _ in range(budget):
            cfg = self.sample_next()
            res = objective(cfg)
            scored = {**res, "score": score_fn(res)}
            self.update_model(cfg, scored)
            results.append({**scored, "config": cfg})
        return results
