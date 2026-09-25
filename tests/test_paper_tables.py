from __future__ import annotations

import pandas as pd

from edge_ai_compression.analysis.paper_tables import escape, fmt_ci, write_paper_tables
from edge_ai_compression.analysis.reproduce import reproduce
from tests.synthetic_db import write_early_prediction_db


def test_escape_and_format():
    assert escape("rtn:w4afp:g32_x&y%") == r"rtn:w4afp:g32\_x\&y\%"
    assert fmt_ci(0.91234, 0.9, 0.93) == "0.912 [0.9, 0.93]"
    assert fmt_ci(0.5, float("nan"), float("nan")) == "0.5"


def test_missing_inputs_become_pending(tmp_path):
    status = write_paper_tables(tmp_path / "nodb", tmp_path / "fig", tmp_path / "gen")
    assert not any(status.values())
    for name in status:
        assert (tmp_path / "gen" / f"{name}.tex").read_text().startswith("\\pending{")


def test_ladder_and_early_prediction_tables_from_data(tmp_path):
    db = write_early_prediction_db(tmp_path / "db")
    rows = []
    for seed in range(3):
        for tag, acc in (("none", 0.95), ("rtn:w8a8:per_channel:minmax", 0.94)):
            rows.append(
                {
                    "model_name": "resnet18_cifar",
                    "dataset": "cifar10",
                    "backend": "edge_quant",
                    "pruning_type": "none",
                    "quantization_type": tag,
                    "accuracy": acc + 0.001 * seed,
                    "accuracy_drop": 0.95 - acc,
                    "latency_median": 3.0,
                    "size_mb": 11.0,
                    "weight_sparsity": 0.0,
                }
            )
    exps = pd.read_csv(db / "experiments.csv")
    pd.concat([exps, pd.DataFrame(rows)]).to_csv(db / "experiments.csv", index=False)
    reproduce(db, tmp_path / "fig", tmp_path / "gen")
    ptq = (tmp_path / "gen" / "ladder_ptq.tex").read_text()
    assert (
        r"\begin{tabular}" in ptq and r"rtn:w8a8:per\_channel:minmax" in ptq and " & 3 \\\\" in ptq
    )
    ep = (tmp_path / "gen" / "early_prediction_w4a8.tex").read_text()
    assert "gbm" in ep and "f=0.1" in ep
    assert (tmp_path / "gen" / "kernel_crossover.tex").read_text().startswith("\\pending{")
