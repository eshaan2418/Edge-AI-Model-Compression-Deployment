from __future__ import annotations

from typing import Callable

import numpy as np


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    n = F.shape[0]
    S = [set() for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(F[p], F[q]):
                S[p].add(q)
            elif _dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F: np.ndarray, front: list[int]) -> np.ndarray:
    if not front:
        return np.array([])
    Ff = F[np.asarray(front)]
    dist = np.zeros(len(front))
    for m in range(F.shape[1]):
        order = np.argsort(Ff[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = Ff[order[-1], m] - Ff[order[0], m]
        if span < 1e-12:
            continue
        for k in range(1, len(order) - 1):
            dist[order[k]] += (Ff[order[k + 1], m] - Ff[order[k - 1], m]) / span
    return dist


def _denormalize(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    out = np.empty_like(x)
    for i, (lo, hi) in enumerate(bounds):
        out[i] = lo + x[i] * (hi - lo)
    return out


def nsga2_minimize(
    evaluate: Callable[[np.ndarray], np.ndarray],
    bounds: list[tuple[float, float]],
    pop_size: int = 20,
    generations: int = 12,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    pop = rng.uniform(0, 1, size=(pop_size, dim))
    objs = np.stack([evaluate(_denormalize(pop[i], bounds)) for i in range(pop_size)])

    for _ in range(generations):
        offspring = []
        for _ in range(pop_size):
            a, b = rng.integers(0, pop_size, size=2)
            eta = rng.uniform(0, 1, size=dim)
            child = eta * pop[a] + (1 - eta) * pop[b]
            child = np.clip(child + rng.normal(0, 0.1, dim), 0, 1)
            offspring.append(child)
        off = np.stack(offspring)
        off_obj = np.stack([evaluate(_denormalize(off[i], bounds)) for i in range(pop_size)])
        comb = np.vstack([pop, off])
        comb_obj = np.vstack([objs, off_obj])
        fronts = fast_non_dominated_sort(comb_obj)
        new_idx: list[int] = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= pop_size:
                new_idx.extend(fr)
            else:
                cd = crowding_distance(comb_obj, fr)
                order = np.argsort(-cd)
                need = pop_size - len(new_idx)
                picked = [fr[j] for j in order[:need]]
                new_idx.extend(picked)
                break
        pop = comb[new_idx]
        objs = comb_obj[new_idx]
    return [(pop[i], objs[i]) for i in range(pop_size)]
