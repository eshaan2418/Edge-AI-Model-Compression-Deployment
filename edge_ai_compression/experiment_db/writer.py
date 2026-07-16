from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from edge_ai_compression.experiment_db.paths import (
    ARTIFACTS_DIR,
    EXPERIMENTS_CSV,
    EXPERIMENTS_JSONL,
)
from edge_ai_compression.experiment_db.record import EXPERIMENT_CSV_FIELDS, ExperimentRecord


def ensure_results_dirs() -> None:
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def append_csv_row(record: ExperimentRecord) -> None:
    ensure_results_dirs()
    path = EXPERIMENTS_CSV
    row = record.to_csv_row()
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EXPERIMENT_CSV_FIELDS))
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_jsonl_line(payload: dict[str, Any]) -> None:
    ensure_results_dirs()
    with open(EXPERIMENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def write_artifacts(
    experiment_id: str,
    *,
    config_dict: dict[str, Any],
    metrics: dict[str, Any],
    model: torch.nn.Module,
    latency_trace_ms: np.ndarray,
    confusion_matrix: np.ndarray | None,
    failure_cases: dict[str, Any] | None,
    tflite_note: str | None = None,
) -> Path:
    ensure_results_dirs()
    root = ARTIFACTS_DIR / experiment_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    with open(root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    torch.save({"model_state": model.state_dict()}, root / "model.pt")

    np.save(root / "latency_trace.npy", latency_trace_ms)
    if confusion_matrix is not None:
        np.save(root / "confusion_matrix.npy", confusion_matrix)
    if failure_cases is not None:
        with open(root / "failure_cases.json", "w", encoding="utf-8") as f:
            json.dump(failure_cases, f, indent=2)

    tflite_path = root / "model.tflite"
    if tflite_note:
        tflite_path.write_text(tflite_note, encoding="utf-8")
    else:
        tflite_path.write_text(
            "TFLite export requires TensorFlow conversion from ONNX or SavedModel; "
            "see legacy quantize_model.py for the Keras/TFLite path.\n",
            encoding="utf-8",
        )

    return root
