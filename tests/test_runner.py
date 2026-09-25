from __future__ import annotations

import csv

import pytest

from edge_ai_compression.core.experiment import ExperimentConfig
from edge_ai_compression.core.runner import ExperimentRunner, load_experiment_config
from edge_ai_compression.experiment_db import artifact_dir, experiments_csv

SMOKE = "edge_ai_compression/configs/experiments/smoke_cpu.yml"
SMOKE_BENCH = "edge_ai_compression/configs/experiments/smoke_benchmark.yml"


def test_smoke_config_logs_to_experiment_db(tmp_path):
    raw = load_experiment_config(SMOKE).to_dict()
    raw["experiment_db"]["results_dir"] = str(tmp_path)
    raw["checkpoint_out"] = str(tmp_path / "model.pt")
    raw["benchmark"]["process_repeats"] = 1
    result = ExperimentRunner(ExperimentConfig.from_dict(raw)).run()

    with open(experiments_csv(tmp_path), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    exp_id = rows[0]["experiment_id"]
    assert result.extras["experiment_id"] == exp_id
    assert (artifact_dir(tmp_path, exp_id) / "fingerprint.json").is_file()
    assert rows[0]["pruning_sparsity"] == "0.5"
    # Only stages that ran are recorded (the default order includes distill/quantize).
    assert rows[0]["compression_order"] == "prune"


def test_smoke_benchmark_config_records_baseline(tmp_path):
    raw = load_experiment_config(SMOKE_BENCH).to_dict()
    raw["experiment_db"]["results_dir"] = str(tmp_path)
    raw["checkpoint_out"] = str(tmp_path / "model.pt")
    ExperimentRunner(ExperimentConfig.from_dict(raw)).run()
    with open(experiments_csv(tmp_path), newline="") as f:
        row = next(csv.DictReader(f))
    assert row["compression_order"] == "baseline"
    assert row["process_repeats"] == "3"
    assert float(row["latency_median_ci_lo"]) <= float(row["latency_median"])


def test_removed_config_keys_raise():
    base = load_experiment_config(SMOKE).to_dict()
    with pytest.raises(ValueError, match="always on"):
        ExperimentConfig.from_dict({**base, "experiment_db": {"enabled": True}})
    with pytest.raises(ValueError, match="renamed"):
        ExperimentConfig.from_dict({**base, "benchmark": {"latency_repeats": 5}})


def test_missing_checkpoint_is_an_error(tmp_path):
    raw = load_experiment_config(SMOKE).to_dict()
    raw["checkpoint_in"] = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError, match="randomly initialized"):
        ExperimentRunner(ExperimentConfig.from_dict(raw)).run()
