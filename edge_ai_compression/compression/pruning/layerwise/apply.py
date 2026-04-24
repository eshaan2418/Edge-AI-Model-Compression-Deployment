from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune

from edge_ai_compression.compression.pruning.layerwise.policy import scores_to_sparsities
from edge_ai_compression.compression.pruning.layerwise.scorers import (
    score_ablation,
    score_activation_variance,
    score_gradient,
    score_magnitude,
)


def _iter_prunable_named(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    out: list[tuple[str, nn.Module, str]] = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            out.append((name, m, "weight"))
    return out


def apply_layerwise_l1_pruning(
    model: nn.Module,
    sparsities: dict[str, float],
) -> nn.Module:
    for name, m, pname in _iter_prunable_named(model):
        if name not in sparsities:
            continue
        prune.l1_unstructured(m, pname, amount=float(sparsities[name]))
    for name, m, pname in _iter_prunable_named(model):
        if name not in sparsities:
            continue
        try:
            prune.remove(m, pname)
        except ValueError:
            pass
    return model


def compute_layerwise_sparsities(
    model: nn.Module,
    scorer: str,
    mean_sparsity: float,
    train_loader,
    device: str,
    baseline_acc: float | None = None,
) -> dict[str, float]:
    scorer = scorer.lower()
    if scorer == "magnitude":
        s = score_magnitude(model)
    elif scorer == "gradient":
        batch = next(iter(train_loader))
        s = score_gradient(model, batch, device)
    elif scorer == "activation":
        s = score_activation_variance(model, train_loader, device)
    elif scorer == "ablation":
        if baseline_acc is None:
            raise ValueError("ablation scorer requires baseline_acc")
        s = score_ablation(model, train_loader, device, baseline_acc)
    else:
        raise ValueError(scorer)
    return scores_to_sparsities(s, mean_sparsity)
