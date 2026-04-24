from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _prunable_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    out: list[tuple[str, nn.Module]] = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            out.append((name, m))
    return out


def score_magnitude(model: nn.Module) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, m in _prunable_modules(model):
        scores[name] = float(m.weight.detach().abs().mean().item())
    return scores


def score_gradient(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: str,
    loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    scores: dict[str, float] = {}
    for name, m in _prunable_modules(model):
        if m.weight.grad is None:
            scores[name] = 0.0
        else:
            g = m.weight.grad
            scores[name] = float((m.weight.detach() * g.detach()).abs().mean().item())
    model.zero_grad(set_to_none=True)
    return scores


def score_activation_variance(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 3,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)

    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            t = out.detach()
            v = t.flatten(1)
            var_b = v.var(dim=1).mean()
            if name not in sums:
                sums[name] = torch.zeros(1)
            sums[name] = sums[name] + var_b.cpu()
            counts[name] += 1

        return hook

    for name, m in _prunable_modules(model):
        hooks.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            model(x)
            if i + 1 >= max_batches:
                break
    for h in hooks:
        h.remove()

    scores: dict[str, float] = {}
    for name in sums:
        scores[name] = float((sums[name] / max(counts[name], 1)).item())
    return scores


def score_ablation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    baseline_acc: float,
    max_batches: int = 2,
) -> dict[str, float]:
    """Layer importance = baseline_acc - acc_after_perturbation (higher = more important)."""
    model.eval()
    importance: dict[str, float] = {}

    def quick_acc(m: nn.Module) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                pred = m(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
                if i + 1 >= max_batches:
                    break
        return correct / max(total, 1)

    backups: dict[str, torch.Tensor] = {}
    for name, m in _prunable_modules(model):
        backups[name] = m.weight.detach().clone()

    for name, m in _prunable_modules(model):
        with torch.no_grad():
            for n2, m2 in _prunable_modules(model):
                m2.weight.copy_(backups[n2])
            m.weight.mul_(0.85)
            acc = quick_acc(model)
            importance[name] = float(max(0.0, baseline_acc - acc))
        with torch.no_grad():
            for n2, m2 in _prunable_modules(model):
                m2.weight.copy_(backups[n2])

    return importance


def get_scorer(name: str) -> str:
    n = name.lower()
    if n not in {"magnitude", "gradient", "activation", "ablation"}:
        raise ValueError(f"Unknown scorer {name}")
    return n
