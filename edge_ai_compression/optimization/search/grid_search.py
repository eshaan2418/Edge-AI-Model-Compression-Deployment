from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable


class GridSearch:
    def __init__(self, space: dict[str, list[Any]]) -> None:
        self.space = space

    def iter_configs(self) -> Iterable[dict[str, Any]]:
        keys = list(self.space.keys())
        for values in product(*(self.space[k] for k in keys)):
            yield dict(zip(keys, values, strict=True))

    def run(self, objective: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cfg in self.iter_configs():
            out.append(objective(cfg))
        return out
