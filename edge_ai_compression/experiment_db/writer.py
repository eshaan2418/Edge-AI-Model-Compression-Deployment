from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from edge_ai_compression.benchmarking.report import BenchmarkReport
from edge_ai_compression.experiment_db.paths import (
    artifact_dir,
    experiments_csv,
    experiments_jsonl,
)
from edge_ai_compression.experiment_db.record import EXPERIMENT_CSV_FIELDS, ExperimentRecord


class SchemaMismatchError(RuntimeError):
    pass


def append_csv_row(record: ExperimentRecord, results_dir: Path) -> None:
    """Append one row; refuse to append to a CSV written with a different schema."""
    path = experiments_csv(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != EXPERIMENT_CSV_FIELDS:
            raise SchemaMismatchError(
                f"{path} was written with a different schema. Move it aside (old results "
                "used the pre-v2 benchmark harness) or point experiment_db.results_dir "
                "somewhere else."
            )
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EXPERIMENT_CSV_FIELDS))
        if write_header:
            w.writeheader()
        w.writerow(record.to_csv_row())


def append_jsonl_line(payload: dict[str, Any], results_dir: Path) -> None:
    path = experiments_jsonl(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def write_artifacts(
    experiment_id: str,
    results_dir: Path,
    *,
    config_dict: dict[str, Any],
    metrics: dict[str, Any],
    model: torch.nn.Module,
    report: BenchmarkReport,
    confusion_matrix: np.ndarray | None,
    failure_cases: dict[str, Any] | None,
) -> Path:
    root = artifact_dir(results_dir, experiment_id)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    with open(root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(root / "fingerprint.json", "w", encoding="utf-8") as f:
        json.dump(report.fingerprint.to_dict(), f, indent=2, default=str)

    torch.save({"model_state": model.state_dict()}, root / "model.pt")

    np.savez_compressed(
        root / "latency_traces_ns.npz",
        **{f"process_{i}": t for i, t in enumerate(report.traces_ns)},
    )
    if confusion_matrix is not None:
        np.save(root / "confusion_matrix.npy", confusion_matrix)
    if failure_cases is not None:
        with open(root / "failure_cases.json", "w", encoding="utf-8") as f:
            json.dump(failure_cases, f, indent=2)

    return root
