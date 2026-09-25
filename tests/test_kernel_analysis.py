from __future__ import annotations

import json

import numpy as np
import pytest

from edge_ai_compression.analysis.kernel_study import crossover, load_rows, sparsity_table


def _write(tmp_path, rows):
    with open(tmp_path / "kernel_benchmarks.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(op, sparsity, medians, fp="a", study="s"):
    return {
        "study": study,
        "fingerprint_hash": fp,
        "op": op,
        "isa": "neon",
        "m": 8,
        "n": 16,
        "k": 32,
        "sparsity": sparsity,
        "latency_median_us": float(np.median(medians)),
        "process_medians_us": medians,
    }


def test_sparsity_table_and_crossover(tmp_path):
    dense = [10.0, 10.2, 9.9, 10.1, 10.0]
    _write(
        tmp_path,
        [
            _row("gemm_f32", 0.0, dense),
            _row("gemm_csr", 0.5, [15.0, 15.2, 14.9, 15.1, 15.0]),  # slower
            _row("gemm_csr", 0.9, [5.0, 5.1, 4.9, 5.0, 5.2]),  # clearly faster
            _row("gemm_sparse24", 0.5, [8.0, 8.1, 7.9, 8.0, 8.2]),
        ],
    )
    table = sparsity_table(load_rows(tmp_path, "s"))
    csr = table[table["op"] == "gemm_csr"].set_index("sparsity")
    assert csr.loc[0.5, "ratio"] == pytest.approx(1.5, rel=0.02)
    assert csr.loc[0.9, "ratio_ci_hi"] < 1.0
    cross = crossover(table)
    assert cross["crossover_sparsity"].iloc[0] == 0.9


def test_load_rows_rejects_mixed_machines(tmp_path):
    _write(
        tmp_path,
        [_row("gemm_f32", 0.0, [1.0, 1.0], fp="a"), _row("gemm_csr", 0.5, [1.0, 1.0], fp="b")],
    )
    with pytest.raises(ValueError, match="machines"):
        load_rows(tmp_path, "s")
    with pytest.raises(ValueError, match="no rows"):
        load_rows(tmp_path, "other")
