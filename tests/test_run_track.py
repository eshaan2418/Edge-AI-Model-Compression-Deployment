from __future__ import annotations

import csv

import pytest

from edge_ai_compression.experiment_db.merge import merge
from edge_ai_compression.experiment_db.writer import SchemaMismatchError
from edge_ai_compression.experiments.run_track import run
from edge_ai_compression.inference import kernels


@pytest.mark.skipif(not kernels.available() and not kernels.kernels_required(), reason="no kernels")
def test_ptq_track_smoke_trains_then_runs_ladder_and_merges(tmp_path):
    results, models = tmp_path / "results", tmp_path / "models"
    run("resnet18_ptq", [0], "cpu", results, models, smoke=True, skip_train=False)
    assert (models / "resnet18_cifar10_seed0.pt").is_file()
    with open(results / "experiments.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    tags = [r["quantization_type"] for r in rows]
    assert tags[0] == "none" and any(t.startswith("adaround") for t in tags)
    assert {r["backend"] for r in rows} == {"edge_f32", "edge_quant"}

    # Merge into an empty DB, then again: second merge adds nothing.
    local = tmp_path / "local"
    first = merge(results, local)
    assert first["experiments.csv"] == len(rows) and first["training_runs.csv"] == 1
    assert merge(results, local)["experiments.csv"] == 0
    with open(local / "experiments.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def test_merge_rejects_schema_mismatch(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "experiments.csv").write_text("experiment_id,x\n1,2\n")
    (tmp_path / "b" / "experiments.csv").write_text("experiment_id,y\n3,4\n")
    with pytest.raises(SchemaMismatchError):
        merge(tmp_path / "a", tmp_path / "b")


@pytest.mark.skipif(not kernels.available() and not kernels.kernels_required(), reason="no kernels")
def test_prune_track_smoke_reuses_checkpoint_and_records_sparsity(tmp_path):
    results, models = tmp_path / "results", tmp_path / "models"
    run("resnet18_ptq", [0], "cpu", results, models, smoke=True, skip_train=False)
    run("resnet18_prune", [0], "cpu", results, models, smoke=True, skip_train=False)
    with open(results / "training_runs.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1  # checkpoint reused, not retrained
    with open(results / "experiments.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["pruning_type"] != "none"]
    assert rows and all(float(r["weight_sparsity"]) > 0.4 for r in rows)
    assert any(r["pruning_type"].endswith(("+finetune", "+lora")) for r in rows)
