from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch.nn as nn

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.fingerprint import collect
from edge_ai_compression.benchmarking.isolation import ProcessResult
from edge_ai_compression.benchmarking.report import BenchmarkReport
from edge_ai_compression.experiment_db import (
    EXPERIMENT_CSV_FIELDS,
    SCHEMA_VERSION,
    SchemaMismatchError,
    append_csv_row,
    experiments_csv,
    record_from_run,
    write_artifacts,
)


def _report() -> BenchmarkReport:
    runs = [
        ProcessResult(
            trace_ns=np.array([1_000_000, 1_100_000, 1_200_000], dtype=np.int64) * (i + 1),
            stages_ms={
                "startup": 1.0,
                "import": 2.0,
                "load": 3.0,
                "first_inference": 4.0,
                "cold_start": 10.0,
            },
            peak_rss_mib=300.0,
            model_peak_rss_mib=100.0,
            energy_j_per_inf=None,
            num_threads=1,
        )
        for i in range(2)
    ]
    return BenchmarkReport.from_runs(
        runs,
        accuracy=0.85,
        val_loss=0.5,
        input_shape=(1, 3, 32, 32),
        config=BenchmarkConfig(process_repeats=2, strict_environment=False),
        size_mb=2.0,
        fingerprint=collect(),
        environment_problems=[],
    )


def _record(report: BenchmarkReport):
    return record_from_run(
        experiment_id="e1",
        model_name="m",
        dataset="cifar10",
        num_params=100,
        flops=1e6,
        compression_order="prune",
        pruning_type="global",
        pruning_sparsity=0.3,
        weight_sparsity=0.29,
        quantization_type="none",
        distillation_enabled=False,
        temperature=4.0,
        alpha=0.5,
        device="cpu",
        baseline_accuracy=0.9,
        report=report,
    )


def test_record_columns_from_report():
    rec = _record(_report())
    row = rec.to_csv_row()
    assert tuple(row) == EXPERIMENT_CSV_FIELDS
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["accuracy_drop"] == pytest.approx(0.05)
    # Process medians are 1.1 ms and 2.2 ms.
    assert row["latency_median"] == pytest.approx(1.65)
    assert row["latency_median_ci_lo"] <= 1.65 <= row["latency_median_ci_hi"]
    assert row["latency_max"] == pytest.approx(2.4)
    assert row["n_iters"] == 6 and row["process_repeats"] == 2
    assert row["energy_j_per_inf"] is None and row["energy_meter"] == "none"
    assert len(row["fingerprint_hash"]) == 16


def test_csv_append_and_schema_mismatch(tmp_path):
    rec = _record(_report())
    append_csv_row(rec, tmp_path)
    append_csv_row(rec, tmp_path)
    with open(experiments_csv(tmp_path), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["experiment_id"] == "e1"

    old = tmp_path / "old"
    old.mkdir()
    experiments_csv(old).write_text("experiment_id,latency_mean,ram_mb,energy_proxy\n")
    with pytest.raises(SchemaMismatchError):
        append_csv_row(rec, old)


def test_write_artifacts(tmp_path):
    report = _report()
    root = write_artifacts(
        "e1",
        tmp_path,
        config_dict={"a": 1},
        metrics={"acc": 0.5},
        model=nn.Linear(4, 2),
        report=report,
        confusion_matrix=np.eye(2, dtype=np.int64),
        failure_cases={"cases": []},
    )
    for name in (
        "config.yaml",
        "metrics.json",
        "fingerprint.json",
        "model.pt",
        "latency_traces_ns.npz",
        "confusion_matrix.npy",
        "failure_cases.json",
    ):
        assert (root / name).is_file(), name
    assert not (root / "model.tflite").exists()
    traces = np.load(root / "latency_traces_ns.npz")
    assert sorted(traces.files) == ["process_0", "process_1"]
    fp = json.loads((root / "fingerprint.json").read_text())
    assert fp["hash"] == report.fingerprint.hash
