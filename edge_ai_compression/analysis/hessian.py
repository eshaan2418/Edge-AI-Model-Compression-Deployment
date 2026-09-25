"""Hutchinson estimates of per-layer Hessian traces.

tr(H) = E_v[v^T H v] for Rademacher v (Hutchinson 1989). One Hessian-vector
product via double backprop gives v_i^T (H v)_i for every parameter block i at
once, i.e. the trace of each diagonal block of the loss Hessian. Used by HAWQ
bit allocation (Phase 3) and as a compressibility signal during training
(Phase 5; Yao et al., PyHessian 2020).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def hutchinson_traces(
    loss_fn: Callable[[], torch.Tensor],
    params: dict[str, torch.Tensor],
    *,
    max_iters: int = 100,
    tol: float = 1e-3,
    min_iters: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Trace of each parameter's Hessian block of ``loss_fn()``.

    Iterates until every running mean changes by less than ``tol`` (relative)
    between iterations, after at least ``min_iters``, or ``max_iters``.
    """
    names = list(params)
    tensors = [params[n] for n in names]
    loss = loss_fn()
    grads = torch.autograd.grad(loss, tensors, create_graph=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sums = [0.0] * len(names)
    means = [0.0] * len(names)
    for it in range(1, max_iters + 1):
        vs = [
            (torch.randint(0, 2, t.shape, generator=gen, dtype=t.dtype) * 2 - 1).to(t.device)
            for t in tensors
        ]
        hvs = torch.autograd.grad(grads, tensors, grad_outputs=vs, retain_graph=True)
        new_means = []
        for i, (v, hv) in enumerate(zip(vs, hvs, strict=True)):
            sums[i] += float((v * hv).sum())
            new_means.append(sums[i] / it)
        converged = it >= min_iters and all(
            abs(n - o) <= tol * (abs(o) + 1e-12) for n, o in zip(new_means, means, strict=True)
        )
        means = new_means
        if converged:
            break
    return dict(zip(names, means, strict=True))


def layer_traces(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    layer_names: list[str],
    **kwargs: float,
) -> dict[str, float]:
    """Hessian-block traces of the cross-entropy loss w.r.t. each named layer's weight."""
    model.eval()
    params = {n: model.get_submodule(n).weight for n in layer_names}

    def loss() -> torch.Tensor:
        return nn.functional.cross_entropy(model(x), y)

    return hutchinson_traces(loss, params, **kwargs)  # type: ignore[arg-type]
