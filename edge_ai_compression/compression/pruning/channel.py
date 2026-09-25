"""Structured channel pruning of residual-block inner channels.

Only channels internal to a residual block are removed (BasicBlock: conv1's
outputs / conv2's inputs; Bottleneck: conv1 and conv2 outputs), so residual
additions keep their shapes. The model is physically rebuilt with smaller dense
layers, so the saving shows up on every backend. Channels are ranked by the L1
norm of their filters (Li et al., ICLR 2017) or by |BN gamma| (network
slimming, Liu et al., ICCV 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn

CRITERIA = ("l1", "bn_gamma")


def _keep_indices(
    conv: nn.Conv2d, bn: nn.BatchNorm2d, amount: float, criterion: str
) -> torch.Tensor:
    if criterion == "l1":
        scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    elif criterion == "bn_gamma":
        scores = bn.weight.detach().abs()
    else:
        raise ValueError(f"channel criterion must be one of {CRITERIA}")
    n = scores.numel()
    keep = max(1, int(round((1.0 - amount) * n)))
    return scores.topk(keep).indices.sort().values


def _slice_conv(
    conv: nn.Conv2d, out_idx: torch.Tensor | None, in_idx: torch.Tensor | None
) -> nn.Conv2d:
    w = conv.weight.detach()
    if out_idx is not None:
        w = w[out_idx]
    if in_idx is not None:
        w = w[:, in_idx]
    new = nn.Conv2d(
        w.shape[1],
        w.shape[0],
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        bias=conv.bias is not None,
    )
    new.weight.data.copy_(w)
    if conv.bias is not None:
        b = conv.bias.detach()
        new.bias.data.copy_(b[out_idx] if out_idx is not None else b)
    return new.train(conv.training)


def _slice_bn(bn: nn.BatchNorm2d, idx: torch.Tensor) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(len(idx), eps=bn.eps, momentum=bn.momentum)
    for name in ("weight", "bias"):
        getattr(new, name).data.copy_(getattr(bn, name).detach()[idx])
    for name in ("running_mean", "running_var"):
        getattr(new, name).copy_(getattr(bn, name)[idx])
    new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new.train(bn.training)  # a fresh module starts in train mode


def _pair(block: nn.Module, a: str, bn_a: str, b: str, amount: float, criterion: str) -> None:
    conv_a, bn, conv_b = getattr(block, a), getattr(block, bn_a), getattr(block, b)
    idx = _keep_indices(conv_a, bn, amount, criterion)
    setattr(block, a, _slice_conv(conv_a, idx, None))
    setattr(block, bn_a, _slice_bn(bn, idx))
    setattr(block, b, _slice_conv(conv_b, None, idx))


def prune_block_channels(model: nn.Module, amount: float, criterion: str = "l1") -> int:
    """Remove ``amount`` of every residual block's inner channels (in place).

    Returns the number of blocks pruned. Supports torchvision BasicBlock and
    Bottleneck; models without such blocks raise, rather than silently doing nothing.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("channel amount must be in [0, 1)")
    count = 0
    for block in model.modules():
        kind = type(block).__name__
        if kind == "BasicBlock":
            _pair(block, "conv1", "bn1", "conv2", amount, criterion)
        elif kind == "Bottleneck":
            _pair(block, "conv1", "bn1", "conv2", amount, criterion)
            _pair(block, "conv2", "bn2", "conv3", amount, criterion)
        else:
            continue
        count += 1
    if count == 0:
        raise ValueError("channel pruning found no BasicBlock/Bottleneck residual blocks")
    return count
