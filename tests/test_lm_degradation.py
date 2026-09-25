from __future__ import annotations

import numpy as np
import pandas as pd

from edge_ai_compression.analysis.lm_degradation import degradation_table, plot_degradation


def _write(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    # Planted: at w4a16_g32, perplexity rises 5% but constraint rate halves.
    for seed in range(3):
        for method, ppl_mult, rate_mult in (
            ("fp", 1.0, 1.0),
            ("w8a8", 1.01, 0.98),
            ("w4a16_g32", 1.05, 0.5),
        ):
            rows.append(
                {
                    "model": "gpt_10m",
                    "seed": seed,
                    "stage": "dpo",
                    "method": method,
                    "perplexity": 3.0 * ppl_mult * (1 + 0.005 * rng.standard_normal()),
                    "constraint_rate": 0.8 * rate_mult * (1 + 0.01 * rng.standard_normal()),
                }
            )
        rows.append(
            {
                "model": "gpt_10m",
                "seed": seed,
                "stage": "pretrain",
                "method": "speculative",
                "perplexity": None,
                "constraint_rate": None,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "lm_evals.csv", index=False)


def test_detects_behavior_degrading_before_capability(tmp_path):
    _write(tmp_path)
    t = degradation_table(tmp_path).set_index("method")
    assert t.loc["fp", "rel_capability"] == 1.0 and t.loc["fp", "rel_behavior"] == 1.0
    assert t.loc["w4a16_g32", "behavior_degrades_first"]
    assert t.loc["w4a16_g32", "gap"] > 0.3
    assert list(t.index) == ["fp", "w8a8", "w4a16_g32"]  # ladder order
    assert plot_degradation(t.reset_index(), tmp_path / "d.png").is_file()
