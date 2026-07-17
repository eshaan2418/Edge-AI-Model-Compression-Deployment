"""Per-layer sensitivity analysis for compression planning.

    python -m edge_ai_compression.analysis.layer_sensitivity \
        --model resnet18_cifar --method prune --amount 0.5 \
        --out reports/sensitivity.md

Uniform compression treats every layer the same, but layers are not equally
forgiving: pruning or quantizing an early feature extractor can wreck the output
while the same operation on a wide late layer barely moves it. This tool probes
each prunable layer *independently* — perturbing only that layer and measuring how
much the model's output drifts on a fixed synthetic input — and ranks layers by
sensitivity. The ranking is what you'd feed a non-uniform / mixed-precision budget.

Honesty: this measures *numerical output drift* (relative L2 and cosine change),
not task accuracy. A layer with low drift is a safer compression target, but only
a real evaluation on real data confirms the accuracy impact.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.utils.reproducibility import set_seed

_PRUNABLE = (nn.Conv2d, nn.Linear)


def prunable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Named Conv2d / Linear layers that carry a weight tensor."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, _PRUNABLE) and getattr(mod, "weight", None) is not None
    ]


def _prune_weight(weight: torch.Tensor, amount: float) -> torch.Tensor:
    """Zero the smallest-magnitude ``amount`` fraction of ``weight`` (magnitude prune)."""
    if amount <= 0.0:
        return weight.clone()
    flat = weight.abs().flatten()
    k = int(amount * flat.numel())
    if k <= 0:
        return weight.clone()
    threshold = torch.kthvalue(flat, k).values
    mask = weight.abs() > threshold
    return weight * mask


def _quantize_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor uniform ``bits``-bit quantization round-trip (fake-quant)."""
    w_min = float(weight.min())
    w_max = float(weight.max())
    if w_max == w_min:
        return weight.clone()
    levels = (2**bits) - 1
    scale = (w_max - w_min) / levels
    q = torch.round((weight - w_min) / scale)
    return q * scale + w_min


def _output_drift(base: torch.Tensor, perturbed: torch.Tensor) -> dict[str, float]:
    diff = (perturbed - base).flatten()
    base_norm = base.flatten().norm().item()
    drift_rel_l2 = (diff.norm().item() / base_norm) if base_norm > 0 else 0.0
    cosine = torch.nn.functional.cosine_similarity(
        base.flatten(), perturbed.flatten(), dim=0, eps=1e-12
    ).item()
    return {"drift_rel_l2": float(drift_rel_l2), "cosine": float(cosine)}


def sensitivity_profile(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    *,
    method: str = "prune",
    amount: float = 0.5,
    bits: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rank each prunable layer by the output drift its perturbation causes.

    For each layer we deep-copy the model, perturb only that layer's weight
    (``method`` = "prune" or "quantize"), run one forward pass on a fixed input,
    and compare the output to the unperturbed baseline. Returns rows sorted by
    ``drift_rel_l2`` descending (most sensitive first).
    """
    if method not in {"prune", "quantize"}:
        raise ValueError("method must be 'prune' or 'quantize'")

    set_seed(seed)
    sample = torch.randn(*input_shape)

    model.eval()
    with torch.no_grad():
        base_out = model(sample).detach()

    total_params = sum(p.numel() for p in model.parameters())
    rows: list[dict[str, Any]] = []
    for name, _mod in prunable_layers(model):
        probe = copy.deepcopy(model)
        target = dict(probe.named_modules())[name]
        with torch.no_grad():
            if method == "prune":
                target.weight.copy_(_prune_weight(target.weight.data, amount))
            else:
                target.weight.copy_(_quantize_weight(target.weight.data, bits))
            probe.eval()
            probe_out = probe(sample).detach()

        drift = _output_drift(base_out, probe_out)
        n_weights = int(target.weight.numel())
        rows.append(
            {
                "layer": name,
                "kind": type(target).__name__,
                "num_weights": n_weights,
                "param_fraction": (n_weights / total_params) if total_params else 0.0,
                **drift,
            }
        )

    rows.sort(key=lambda r: r["drift_rel_l2"], reverse=True)
    return rows


def to_markdown(rows: list[dict[str, Any]], *, method: str, amount: float, bits: int) -> str:
    probe = f"prune amount={amount:g}" if method == "prune" else f"quantize bits={bits}"
    lines = [
        "# Layer sensitivity profile",
        "",
        f"- Probe: **{probe}** (applied to one layer at a time)",
        "- Metric: relative L2 output drift on synthetic input (higher = more sensitive).",
        "- ⚠️ Drift is numerical, not accuracy; confirm on real data before deciding budgets.",
        "",
        "| Rank | Layer | Kind | Weights | Param % | Rel L2 drift | Cosine |",
        "| ---- | ----- | ---- | ------- | ------- | ------------ | ------ |",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['layer']}` | {r['kind']} | {r['num_weights']:,} | "
            f"{r['param_fraction'] * 100:.2f}% | {r['drift_rel_l2']:.4f} | {r['cosine']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.analysis.layer_sensitivity",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="resnet18_cifar", help="Registry model name.")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--input-shape", default="1,3,32,32", help="Comma-separated, e.g. 1,3,32,32.")
    p.add_argument("--method", choices=["prune", "quantize"], default="prune")
    p.add_argument("--amount", type=float, default=0.5, help="Prune fraction (prune method).")
    p.add_argument("--bits", type=int, default=8, help="Quantization bits (quantize method).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Optional Markdown/JSON output path.")
    args = p.parse_args(argv)

    if args.model not in ModelRegistry.available():
        print(f"Unknown model '{args.model}'. Available: {', '.join(ModelRegistry.available())}")
        return 1
    try:
        shape = tuple(int(x) for x in args.input_shape.split(","))
    except ValueError:
        print(f"Invalid --input-shape: {args.input_shape}")
        return 1

    model = ModelRegistry.create(args.model, num_classes=args.num_classes)
    rows = sensitivity_profile(
        model, shape, method=args.method, amount=args.amount, bits=args.bits, seed=args.seed
    )

    print(f"Most sensitive layers under {args.method} (top {min(5, len(rows))}):")
    for r in rows[:5]:
        print(
            f"  {r['drift_rel_l2']:.4f}  {r['layer']} ({r['kind']}, {r['num_weights']:,} weights)"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix == ".json":
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        else:
            args.out.write_text(
                to_markdown(rows, method=args.method, amount=args.amount, bits=args.bits),
                encoding="utf-8",
            )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
