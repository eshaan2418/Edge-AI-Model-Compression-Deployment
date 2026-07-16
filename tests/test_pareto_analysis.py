from __future__ import annotations

import pandas as pd

from edge_ai_compression.analysis.pareto import compute_pareto


def _row(eid, acc, lat, size, ram):
    return {
        "experiment_id": eid,
        "accuracy": acc,
        "latency_mean": lat,
        "size_mb": size,
        "ram_mb": ram,
    }


def test_frontier_excludes_dominated_points():
    # B dominates C (better accuracy, lower on every cost). A is a distinct
    # trade-off (highest accuracy but most expensive) so it survives.
    df = pd.DataFrame(
        [
            _row("A", 0.95, 30, 40, 400),
            _row("B", 0.90, 10, 10, 100),
            _row("C", 0.85, 20, 20, 200),
        ]
    )
    frontier, max_obj, min_obj = compute_pareto(df)
    ids = set(frontier["experiment_id"])
    assert "C" not in ids
    assert {"A", "B"} <= ids
    assert max_obj == ("accuracy",)
    assert min_obj == ("latency_mean", "size_mb", "ram_mb")


def test_missing_objective_columns_are_ignored():
    # Only accuracy + size_mb are present; latency/ram objectives are dropped.
    df = pd.DataFrame(
        [
            {"experiment_id": "A", "accuracy": 0.9, "size_mb": 10},
            {"experiment_id": "B", "accuracy": 0.8, "size_mb": 20},
        ]
    )
    frontier, max_obj, min_obj = compute_pareto(df)
    assert max_obj == ("accuracy",)
    assert min_obj == ("size_mb",)
    # A dominates B on both used objectives.
    assert set(frontier["experiment_id"]) == {"A"}


def test_rows_with_null_objectives_are_dropped():
    df = pd.DataFrame(
        [
            _row("A", 0.9, 10, 10, 100),
            _row("B", None, 5, 5, 50),
        ]
    )
    frontier, _, _ = compute_pareto(df)
    assert set(frontier["experiment_id"]) == {"A"}


def test_full_cli_writes_outputs(tmp_path):
    import subprocess
    import sys

    results = tmp_path / "experiments.csv"
    pd.DataFrame([_row("A", 0.9, 10, 10, 100), _row("B", 0.8, 20, 20, 200)]).to_csv(
        results, index=False
    )

    out = tmp_path / "pareto"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "edge_ai_compression.analysis.pareto",
            "--results",
            str(results),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "pareto_frontier.csv").is_file()
    assert (out / "pareto_frontier.md").is_file()
    md = (out / "pareto_frontier.md").read_text()
    assert "Pareto Frontier" in md
