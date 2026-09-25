from __future__ import annotations

import csv

import pytest

from edge_ai_compression.experiments.compress_runs import PANEL, compress_runs, method_tag
from edge_ai_compression.experiments.run_track import run


def test_method_tags_are_distinct():
    tags = {method_tag(m) for m in PANEL}
    assert len(tags) == len(PANEL)
    assert method_tag("w4g32") == ("none", "rtn:w4afp:g32")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("targets")
    results, models = tmp / "results", tmp / "models"
    run("pretrain_variants", [0], "cpu", results, models, smoke=True, skip_train=False)
    return results


def test_compress_runs_links_rows_to_training_runs_and_resumes(trained):
    overrides = {"limit_samples": 16, "batch_size": 8, "num_workers": 0}
    methods = ["w8a8", "nm24"]
    n = compress_runs(trained, methods=methods, overrides=overrides)
    assert n == 2 * 2  # 2 training runs x 2 methods
    assert compress_runs(trained, methods=methods, overrides=overrides) == 0  # resumed
    with open(trained / "training_runs.csv", newline="") as f:
        runs = {r["run_id"]: r for r in csv.DictReader(f)}
    with open(trained / "experiments.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["source_run_id"] for r in rows} == set(runs)
    assert all(r["source_step"] == runs[r["source_run_id"]]["steps"] for r in rows)


def test_all_steps_covers_every_checkpoint(trained):
    overrides = {"limit_samples": 16, "batch_size": 8, "num_workers": 0}
    n = compress_runs(trained, methods=["w8a8"], all_steps=True, overrides=overrides)
    assert n > 0  # intermediate checkpoints added (the final ones were already done)
