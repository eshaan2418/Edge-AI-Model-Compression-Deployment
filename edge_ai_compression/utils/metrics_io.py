"""Load and normalize metrics from the toolkit's various JSON report shapes.

The demo, benchmark harness, and experiment runner each emit slightly different
JSON. This module turns any of them into one canonical metrics dict so the
reporting / hardware / model-card tools don't each re-implement the parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict. Raises FileNotFoundError if absent."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def _latency_mean(block: Any) -> float | None:
    """Latency mean in ms from either a nested block or a scalar."""
    if isinstance(block, dict):
        val = block.get("mean")
        return float(val) if val is not None else None
    if isinstance(block, (int, float)):
        return float(block)
    return None


def _latency_percentiles(block: Any) -> dict[str, float]:
    if not isinstance(block, dict):
        return {}
    return {k: float(block[k]) for k in ("p50", "p95", "p99") if block.get(k) is not None}


def extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a report/metrics dict into canonical fields.

    Handles three shapes:
    * demo report      — has a ``compressed`` sub-report; that is used.
    * benchmark report — top-level ``latency_ms`` block + size/params.
    * plain metrics    — flat keys, ``latency_ms`` may be a scalar.

    Returns keys: ``source_kind``, ``accuracy`` (+ ``accuracy_is_synthetic``),
    ``latency_ms``, ``latency_percentiles``, ``size_mb``, ``ram_mb``,
    ``num_parameters``, ``flops``. Missing values are ``None``.
    """
    # Demo report: drill into the compressed model's metrics.
    if "compressed" in data and isinstance(data["compressed"], dict):
        block = data["compressed"]
        source_kind = "demo"
        accuracy = block.get("synthetic_accuracy")
        accuracy_is_synthetic = True
    else:
        block = data
        source_kind = "benchmark" if isinstance(data.get("latency_ms"), dict) else "metrics"
        accuracy = data.get("accuracy")
        accuracy_is_synthetic = "synthetic_accuracy" in data
        if accuracy is None:
            accuracy = data.get("synthetic_accuracy")

    ram = block.get("peak_ram_mib")
    if ram is None:
        ram = block.get("ram_mb")

    return {
        "source_kind": source_kind,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "accuracy_is_synthetic": bool(accuracy_is_synthetic),
        "latency_ms": _latency_mean(block.get("latency_ms")),
        "latency_percentiles": _latency_percentiles(block.get("latency_ms")),
        "size_mb": float(block["size_mb"]) if block.get("size_mb") is not None else None,
        "ram_mb": float(ram) if ram is not None else None,
        "num_parameters": (
            int(block["num_parameters"]) if block.get("num_parameters") is not None else None
        ),
        "flops": (
            float(block["flops_estimate"]) if block.get("flops_estimate") is not None else None
        ),
    }
