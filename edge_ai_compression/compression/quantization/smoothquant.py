"""SmoothQuant (Xiao et al., ICML 2023) and activation-outlier measurement.

Per input channel j of the Linear layers fed by a LayerNorm:
    s_j = max|X_j|^alpha / max|W_{:,j}|^(1 - alpha)
The LayerNorm's affine weight and bias are divided by s (so X' = X / s) and the
consuming Linear weights' columns multiplied by s (W' = W * s). The float
function is unchanged; activation outliers migrate into the weights, which
per-channel weight quantization absorbs.

Whether it helps depends on outliers existing. Dettmers et al. (LLM.int8(),
2022) report systematic outlier features emerging at the billions-of-parameters
scale, so small CIFAR ViTs may have none; ``outlier_stats`` measures this before
any claim is made (DECISIONS D3.8).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.compression.quantization.calibration import calibration_batches


@dataclass(frozen=True)
class SmoothQuantConfig:
    alpha: float = 0.5
    num_samples: int = 128

    @staticmethod
    def from_dict(d: dict) -> SmoothQuantConfig:
        unknown = set(d) - {"alpha", "num_samples"}
        if unknown:
            raise ValueError(f"unknown smoothquant options: {sorted(unknown)}")
        return SmoothQuantConfig(**d)


def norm_linear_groups(model: nn.Module) -> dict[str, list[str]]:
    """{LayerNorm name: names of the Linear layers consuming its output directly}."""
    gm = fx.symbolic_trace(model)
    groups: dict[str, list[str]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(
            gm.get_submodule(str(node.target)), nn.LayerNorm
        ):
            continue
        # Shape reads (x.shape / x.size()) are not value consumers.
        users = [
            u
            for u in node.users
            if not (u.op == "call_function" and u.target is getattr)
            and not (u.op == "call_method" and u.target in ("size", "dim"))
        ]
        if users and all(
            u.op == "call_module" and isinstance(gm.get_submodule(str(u.target)), nn.Linear)
            for u in users
        ):
            groups[str(node.target)] = [str(u.target) for u in users]
    return groups


@torch.no_grad()
def channel_absmax(
    model: nn.Module, loader: DataLoader, layer_names: list[str], num_samples: int, device: str
) -> dict[str, torch.Tensor]:
    """Per-input-channel max |x| at each named layer's input over calibration data."""
    stats: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def fn(_m: nn.Module, args: tuple) -> None:
            x = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(dim=0)
            stats[name] = torch.maximum(stats[name], x) if name in stats else x

        return fn

    hooks = [model.get_submodule(n).register_forward_pre_hook(hook(n)) for n in layer_names]
    model.eval().to(device)
    try:
        for x in calibration_batches(loader, num_samples, device):
            model(x)
    finally:
        for h in hooks:
            h.remove()
    return stats


def outlier_stats(
    model: nn.Module, loader: DataLoader, num_samples: int = 128, device: str = "cpu"
) -> dict[str, float]:
    """Max / median of per-channel |activation| maxima at each LayerNorm-fed Linear input.

    Ratios near 1-10 mean no systematic outlier channels; LLM-scale outliers give
    ratios in the tens to hundreds.
    """
    groups = norm_linear_groups(model)
    first_users = [users[0] for users in groups.values()]
    stats = channel_absmax(model, loader, first_users, num_samples, device)
    return {n: float(s.max() / s.median().clamp_min(1e-12)) for n, s in stats.items()}


@torch.no_grad()
def smooth(model: nn.Module, loader: DataLoader, cfg: SmoothQuantConfig, device: str) -> int:
    """Apply SmoothQuant in place to every LayerNorm -> Linear group. Returns #groups."""
    groups = norm_linear_groups(model)
    first_users = [users[0] for users in groups.values()]
    act = channel_absmax(model, loader, first_users, cfg.num_samples, device)
    for (norm_name, users), first in zip(groups.items(), first_users, strict=True):
        norm = model.get_submodule(norm_name)
        linears = [model.get_submodule(u) for u in users]
        w_max = torch.stack([lin.weight.abs().amax(dim=0) for lin in linears]).amax(dim=0)
        s = act[first].clamp_min(1e-5).pow(cfg.alpha) / w_max.clamp_min(1e-5).pow(1 - cfg.alpha)
        s = s.clamp_min(1e-5)
        norm.weight.div_(s)
        norm.bias.div_(s)
        for lin in linears:
            lin.weight.mul_(s)
    return len(groups)
