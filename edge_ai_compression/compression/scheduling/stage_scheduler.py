from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class StageSpec:
    name: str
    enabled: bool


class StageScheduler:
    """Declarative ordering for compression stages (for future DAG execution)."""

    def __init__(self, stages: list[StageSpec]) -> None:
        self.stages = stages

    def ordered_enabled(self) -> list[str]:
        return [s.name for s in self.stages if s.enabled]

    def run_hooks(self, hooks: dict[str, Callable[[], None]]) -> None:
        for name in self.ordered_enabled():
            if name in hooks:
                hooks[name]()
