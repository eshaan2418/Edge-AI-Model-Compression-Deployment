from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge_ai_compression.analysis.latency_proxy import (
    KernelLatencyModel,
    naive_correlations,
    proxy_comparison,
)

# Planted per-kernel cost (us) = c * (M N K)^0.9 * (1 - s)^0.6; CSR is slowest per FLOP.
COST = {"gemm_f32": 1e-3, "gemm_s8": 4e-4, "gemm_csr": 3e-3}
BACKEND = {"edge_f32": "gemm_f32", "edge_int8": "gemm_s8", "edge_csr": "gemm_csr"}


def _t(op, m, n, k, s):
    return COST[op] * (m * n * k) ** 0.9 * (1 - s) ** 0.6


def _kernels():
    rows = []
    for op in COST:
        for m in (16, 64, 256):
            for n in (1, 64, 1024):
                for k in (27, 256, 2304):
                    for s in (0.0,) if op != "gemm_csr" else (0.5, 0.8, 0.95):
                        rows.append(
                            {
                                "op": op,
                                "m": m,
                                "n": n,
                                "k": k,
                                "sparsity": s,
                                "latency_median_us": _t(op, m, n, k, s),
                                "fingerprint_hash": "fp",
                            }
                        )
    return pd.DataFrame(rows)


def _models():
    rng = np.random.default_rng(0)
    exps, shapes = [], {}
    for width in (0.25, 0.5, 1.0, 2.0):
        layers = [{"m": int(64 * width), "n": 1024, "k": 27}] + [
            {"m": int(c * width), "n": n, "k": int(c * width) * 9}
            for c, n in ((64, 1024), (128, 256), (256, 64), (512, 16))
        ]
        flops = sum(layer["m"] * layer["n"] * layer["k"] for layer in layers)
        params = sum(layer["m"] * layer["k"] for layer in layers)
        for backend, sp in (
            ("edge_f32", 0.0),
            ("edge_int8", 0.0),
            ("edge_csr", 0.5),
            ("edge_csr", 0.8),
        ):
            eid = f"w{width}-{backend}-{sp}"
            op = BACKEND[backend]
            lat_us = sum(_t(op, layer["m"], layer["n"], layer["k"], sp) for layer in layers)
            exps.append(
                {
                    "experiment_id": eid,
                    "fingerprint_hash": "fp",
                    "backend": backend,
                    "model_name": f"w{width}",
                    "pruning_type": f"unstr:{sp}",
                    "quantization_type": "none",
                    "weight_sparsity": sp,
                    "flops": flops,
                    "num_params": params,
                    "size_mb": params * (1 - sp) * 4e-6,
                    "latency_median": (lat_us + 50 * len(layers))
                    / 1e3
                    * (1 + 0.01 * rng.standard_normal()),
                }
            )
            shapes[eid] = layers
    return pd.DataFrame(exps), shapes


def test_kernel_model_fits_planted_costs():
    km = KernelLatencyModel().fit(_kernels())
    assert km.predict_us("gemm_f32", 128, 256, 512, 0.0) == pytest.approx(
        _t("gemm_f32", 128, 256, 512, 0.0), rel=0.05
    )
    assert km.predict_us("gemm_csr", 128, 256, 512, 0.7) == pytest.approx(
        _t("gemm_csr", 128, 256, 512, 0.7), rel=0.1
    )
    with pytest.raises(KeyError):
        km.predict_us("gemm_w4", 1, 1, 1, 0.0)


def test_flops_misleads_across_backends_and_learned_proxy_does_not():
    exps, shapes = _models()
    naive = naive_correlations(exps).set_index("backend")
    assert naive.loc["edge_f32", "rho_flops"] == pytest.approx(1.0)  # one kernel: FLOPs ranks fine
    # Pooled across kernels with different cost per FLOP, FLOPs ranks worse than within one.
    assert naive.loc["all backends", "rho_flops"] < naive.loc["edge_f32", "rho_flops"]
    table = proxy_comparison(exps, KernelLatencyModel().fit(_kernels()), shapes).set_index("proxy")
    assert table.loc["learned", "spearman"] > 0.95
    assert table.loc["learned", "spearman"] > table.loc["flops", "spearman"]
    assert table.loc["learned", "mape"] < table.loc["flops", "mape"]
