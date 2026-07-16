from __future__ import annotations

import numpy as np
import torch.nn as nn

import edge_ai_compression.experiment_db.writer as writer_mod
from edge_ai_compression.experiment_db.record import record_from_run
from edge_ai_compression.experiment_db.writer import append_csv_row, write_artifacts


def test_write_artifact_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_mod, "EXPERIMENTS_CSV", tmp_path / "experiments.csv")
    monkeypatch.setattr(writer_mod, "EXPERIMENTS_JSONL", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(writer_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")

    rec = record_from_run(
        experiment_id="e1",
        model_name="m",
        dataset="cifar10",
        num_params=100,
        flops=1e6,
        compression_order="prune",
        pruning_type="global",
        pruning_sparsity=0.3,
        quantization_type="none",
        distillation_enabled=False,
        temperature=4.0,
        alpha=0.5,
        device="cpu",
        baseline_accuracy=0.9,
        accuracy=0.85,
        latency={"mean": 1.0, "std": 0.1, "p50": 1.0, "p90": 1.2, "p99": 1.5},
        size_mb=2.0,
        ram_mb=100.0,
        energy_proxy=3.0,
    )
    append_csv_row(rec)
    assert writer_mod.EXPERIMENTS_CSV.is_file()

    m = nn.Linear(4, 2)
    root = write_artifacts(
        "e1",
        config_dict={"a": 1},
        metrics={"acc": 0.5},
        model=m,
        latency_trace_ms=np.array([1.0, 1.1]),
        confusion_matrix=np.eye(2, dtype=np.int64),
        failure_cases={"cases": []},
    )
    assert (root / "model.pt").is_file()
    assert (root / "latency_trace.npy").is_file()
