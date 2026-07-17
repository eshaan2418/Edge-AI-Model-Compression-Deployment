"""Export a PyTorch model to a deployable artifact.

Supported formats:

* ``torchscript`` — ``torch.jit`` trace, saved as ``.pt``. Native to PyTorch,
  always available.
* ``onnx`` — ``torch.onnx.export``. Requires the ``onnx`` package (and, for the
  default dynamo exporter, ``onnxscript``); a clear error is raised if missing.
* ``json`` — a human-readable *metadata* summary (architecture, parameter count,
  size, sparsity). This is NOT the weights — it documents the model, it does not
  serialize it for inference.
* ``tflite`` — intentionally unsupported from PyTorch here. TFLite is produced
  from TensorFlow/Keras graphs; converting a PyTorch model requires an
  ONNX -> TensorFlow -> TFLite toolchain. The error explains the options,
  rather than silently emitting a fake ``.tflite``.

Honesty note: no format pretends to do more than it does. In particular the
``json`` export is metadata only, and ``tflite`` refuses rather than faking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from edge_ai_compression.theory.model_complexity import model_summary

SUPPORTED_FORMATS: tuple[str, ...] = ("torchscript", "onnx", "json", "tflite")

_DEFAULT_INPUT_SHAPE = (1, 3, 32, 32)


class ExportError(RuntimeError):
    """Raised when an export cannot be completed (bad format, missing dep, etc.)."""


def _example_input(input_shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*input_shape)


def _export_torchscript(model: nn.Module, out_path: Path, input_shape: tuple[int, ...]) -> None:
    model.eval()
    example = _example_input(input_shape)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
    scripted.save(str(out_path))


def _export_onnx(model: nn.Module, out_path: Path, input_shape: tuple[int, ...]) -> None:
    model.eval()
    example = _example_input(input_shape)
    try:
        torch.onnx.export(
            model,
            example,
            str(out_path),
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
    except Exception as exc:  # torch raises OnnxExporterError / ImportError variants
        msg = str(exc)
        if "onnx" in msg.lower() and ("not installed" in msg.lower() or "no module" in msg.lower()):
            raise ExportError(
                "ONNX export requires the 'onnx' package. Install it with "
                "`pip install onnx` (and `pip install onnxscript` for the default "
                "torch.export-based exporter)."
            ) from exc
        raise ExportError(f"ONNX export failed: {exc}") from exc


def _export_json(model: nn.Module, out_path: Path, input_shape: tuple[int, ...]) -> None:
    summary = model_summary(model)
    summary["export_kind"] = "metadata_only"
    summary["input_shape"] = list(input_shape)
    summary["note"] = (
        "This file documents the model architecture and size. It does NOT "
        "contain weights and cannot be used for inference."
    )
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _export_tflite(model: nn.Module, out_path: Path, input_shape: tuple[int, ...]) -> None:
    raise ExportError(
        "Direct PyTorch -> TFLite export is not supported. TFLite is produced "
        "from TensorFlow/Keras graphs. Options:\n"
        "  1. Use the TensorFlow root scripts in this repo, which export real "
        ".tflite models from Keras.\n"
        "  2. Export to ONNX here, then convert ONNX -> TF -> TFLite with a "
        "tool such as onnx2tf / onnx-tf.\n"
        "Refusing rather than emitting a placeholder .tflite."
    )


_EXPORTERS = {
    "torchscript": _export_torchscript,
    "onnx": _export_onnx,
    "json": _export_json,
    "tflite": _export_tflite,
}

_DEFAULT_EXTENSION = {
    "torchscript": ".pt",
    "onnx": ".onnx",
    "json": ".json",
    "tflite": ".tflite",
}


def export_model(
    model: nn.Module,
    out_path: str | Path,
    fmt: str,
    *,
    input_shape: tuple[int, ...] = _DEFAULT_INPUT_SHAPE,
) -> str:
    """Export ``model`` to ``out_path`` in ``fmt``; return the path written.

    Raises ``ExportError`` for unknown formats or missing optional dependencies.
    """
    fmt = fmt.lower()
    if fmt not in _EXPORTERS:
        raise ExportError(
            f"Unknown export format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}."
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _EXPORTERS[fmt](model, out, tuple(input_shape))
    return str(out)


def default_extension(fmt: str) -> str:
    return _DEFAULT_EXTENSION.get(fmt.lower(), "")


def resolve_output_path(out_path: str | Path | None, fmt: str, model_name: str) -> Path:
    """Pick an output path, defaulting to ``exports/<model>.<ext>`` when unset."""
    if out_path is not None:
        return Path(out_path)
    return Path("exports") / f"{model_name}{default_extension(fmt)}"


def _cli_args(argv: list[str] | None = None) -> Any:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.export",
        description="Export a registered model to a deployable format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Name of a model in ModelRegistry.")
    p.add_argument(
        "--format",
        required=True,
        choices=SUPPORTED_FORMATS,
        help="Export format.",
    )
    p.add_argument("--num-classes", type=int, default=10, help="Number of output classes.")
    p.add_argument("--out", default=None, help="Output path (default: exports/<model>.<ext>).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from edge_ai_compression.core.registry import ModelRegistry

    args = _cli_args(argv)
    model = ModelRegistry.create(args.model, num_classes=args.num_classes)
    out = resolve_output_path(args.out, args.format, args.model)
    try:
        written = export_model(model, out, args.format)
    except ExportError as exc:
        raise SystemExit(f"Export failed: {exc}") from exc
    print(f"Exported {args.model} as {args.format} -> {written}")


if __name__ == "__main__":
    main()
