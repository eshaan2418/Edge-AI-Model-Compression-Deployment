from __future__ import annotations

import pandas as pd

from edge_ai_compression.analysis.pareto_report import frontier
from edge_ai_compression.analysis.reproduce import reproduce
from tests.synthetic_db import write_early_prediction_db


def test_reproduce_marks_missing_inputs_pending_and_runs_the_rest(tmp_path):
    db = write_early_prediction_db(tmp_path / "db")
    status = reproduce(db, tmp_path / "fig")
    assert status["kernel study + roofline"] == "PENDING"
    assert status["early predictability"] == "ok"  # w4a8 has data; other methods are skipped
    assert (tmp_path / "fig" / "early_prediction_w4a8.png").is_file()
    assert not (tmp_path / "fig" / "early_prediction_nm24.csv").exists()
    manifest = (tmp_path / "fig" / "MANIFEST.md").read_text()
    assert "[PENDING: run configs/studies/kernel_sparsity_m5.yml]" in manifest


def test_pareto_frontier_per_machine():
    exps = pd.DataFrame(
        [
            {
                "experiment_id": "a",
                "fingerprint_hash": "m1",
                "accuracy": 0.9,
                "latency_median": 5.0,
                "size_mb": 10.0,
                "energy_j_per_inf": None,
                "backend": "x",
            },
            {
                "experiment_id": "b",
                "fingerprint_hash": "m1",
                "accuracy": 0.8,
                "latency_median": 6.0,
                "size_mb": 12.0,
                "energy_j_per_inf": None,
                "backend": "x",
            },  # dominated by a
            {
                "experiment_id": "c",
                "fingerprint_hash": "m1",
                "accuracy": 0.7,
                "latency_median": 1.0,
                "size_mb": 3.0,
                "energy_j_per_inf": None,
                "backend": "y",
            },
            {
                "experiment_id": "d",
                "fingerprint_hash": "m2",
                "accuracy": 0.1,
                "latency_median": 99.0,
                "size_mb": 99.0,
                "energy_j_per_inf": None,
                "backend": "x",
            },  # alone on its machine
        ]
    )
    on = frontier(exps).set_index("experiment_id")["on_frontier"]
    assert on.to_dict() == {"a": True, "b": False, "c": True, "d": True}
