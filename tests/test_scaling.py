from __future__ import annotations

import numpy as np
import pytest

from edge_ai_compression.analysis.scaling import (
    fit_power_law,
    plot_scaling,
    scaling_points,
    scaling_table,
)
from tests.synthetic_db import write_scaling_db


def test_fit_recovers_exact_power_law():
    x = np.array([1e5, 3e5, 1e6, 3e6, 1e7])
    fit = fit_power_law(x, 0.3 * (x / 1e5) ** -0.4 + 0.02)
    assert fit["b"] == pytest.approx(0.4, rel=1e-3) and fit["beats_constant"]


def test_scaling_table_recovers_exponent_and_flags_flat_series(tmp_path):
    db = write_scaling_db(tmp_path)
    table = scaling_table(db, "w4a8", "num_params", {"epochs": 30}, n_boot=200).set_index("variant")
    std = table.loc["standard"]
    assert std["b_lo"] <= 0.4 <= std["b_hi"] and std["beats_constant"]
    assert not table.loc["flat"]["beats_constant"]
    pts = scaling_points(db, "w4a8", "num_params", {"epochs": 30})
    assert plot_scaling(pts, table.reset_index(), tmp_path / "s.png", "Parameters").is_file()
