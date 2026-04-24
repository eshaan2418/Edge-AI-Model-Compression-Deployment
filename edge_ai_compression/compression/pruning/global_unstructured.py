from __future__ import annotations

from typing import Iterable

import torch.nn as nn
import torch.nn.utils.prune as prune


def _iter_prunable_modules(model: nn.Module) -> Iterable[tuple[nn.Module, str]]:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            yield module, "weight"
        if isinstance(module, nn.Linear):
            yield module, "weight"


def apply_global_unstructured_pruning(model: nn.Module, amount: float = 0.5) -> nn.Module:
    parameters_to_prune: list[tuple[nn.Module, str]] = list(_iter_prunable_modules(model))
    if not parameters_to_prune:
        return model
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    return model


def remove_pruning_reparametrization(model: nn.Module) -> nn.Module:
    for module, name in _iter_prunable_modules(model):
        try:
            prune.remove(module, name)
        except ValueError:
            pass
    return model
