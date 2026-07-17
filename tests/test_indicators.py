from __future__ import annotations

import json
import math

from edge_ai_compression.optimization.indicators import (
    epsilon_indicator,
    hypervolume,
    indicators,
    main,
    spacing,
)


def test_hypervolume_single_point():
    # Box from (1,1) to reference (3,3) has area 4.
    assert hypervolume([[1.0, 1.0]], [3.0, 3.0]) == 4.0


def test_hypervolume_two_points_hand_verified():
    # Union of [(1,2)-(3,3)] and [(2,1)-(3,3)] = 2 + 2 - 1 overlap = 3.
    hv = hypervolume([[1.0, 2.0], [2.0, 1.0]], [3.0, 3.0])
    assert math.isclose(hv, 3.0)


def test_hypervolume_dominated_point_insensitive():
    front = [[1.0, 2.0], [2.0, 1.0]]
    hv_base = hypervolume(front, [3.0, 3.0])
    # (2.5, 2.5) is dominated by both -> must not change the dominated volume.
    hv_with = hypervolume(front + [[2.5, 2.5]], [3.0, 3.0])
    assert math.isclose(hv_base, hv_with)


def test_hypervolume_increases_with_better_point():
    front = [[2.0, 2.0]]
    hv_base = hypervolume(front, [3.0, 3.0])  # area 1
    hv_better = hypervolume(front + [[1.0, 1.0]], [3.0, 3.0])  # area 4
    assert hv_better > hv_base


def test_hypervolume_3d():
    # Single point (1,1,1) to ref (3,3,3) -> 2*2*2 = 8.
    assert hypervolume([[1.0, 1.0, 1.0]], [3.0, 3.0, 3.0]) == 8.0


def test_epsilon_indicator_zero_when_identical():
    front = [[1.0, 2.0], [2.0, 1.0]]
    assert math.isclose(epsilon_indicator(front, front), 0.0)


def test_epsilon_indicator_positive_when_worse():
    front = [[2.0, 2.0]]
    reference = [[1.0, 1.0]]
    # front must shift by +1 in each dim to dominate reference.
    assert math.isclose(epsilon_indicator(front, reference), 1.0)


def test_spacing_zero_for_single_point():
    assert spacing([[1.0, 1.0]]) == 0.0


def test_indicators_over_dicts():
    points = [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 1.0},
        {"x": 2.5, "y": 2.5},  # dominated
    ]
    rep = indicators(
        points,
        maximize=(),
        minimize=("x", "y"),
        reference={"x": 3.0, "y": 3.0},
    )
    assert rep["num_points"] == 3
    assert rep["front_size"] == 2
    assert math.isclose(rep["hypervolume"], 3.0)


def test_indicators_skips_incomplete_points():
    points = [{"accuracy": 0.9, "latency_ms_mean": 10.0, "size_mb": 5.0, "peak_ram_mib": 20.0}]
    rep = indicators(points)  # defaults require all four keys
    assert rep["num_points"] == 1
    rep_missing = indicators([{"accuracy": 0.9}])
    assert rep_missing["num_points"] == 0


def test_cli_json(tmp_path):
    data = [
        {"accuracy": 0.9, "latency_ms_mean": 10.0, "size_mb": 5.0, "peak_ram_mib": 20.0},
        {"accuracy": 0.85, "latency_ms_mean": 5.0, "size_mb": 3.0, "peak_ram_mib": 15.0},
    ]
    src = tmp_path / "runs.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "ind.json"
    assert main(["--results", str(src), "--out", str(out)]) == 0
    assert "hypervolume" in json.loads(out.read_text(encoding="utf-8"))


def test_cli_missing_file_returns_2(tmp_path):
    assert main(["--results", str(tmp_path / "nope.csv")]) == 2
