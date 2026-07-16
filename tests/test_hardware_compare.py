from __future__ import annotations

import json

from edge_ai_compression.hardware.compare import compare_all_profiles, main, to_markdown


def test_compare_covers_all_profiles():
    metrics = {"accuracy": 0.9, "latency_ms": 10.0, "size_mb": 5.0, "ram_mb": 100.0}
    rows = compare_all_profiles(metrics)
    names = {r["profile"] for r in rows}
    assert names == {"cpu", "raspberry_pi", "smartphone", "microcontroller_sim"}


def test_feasible_sorted_first():
    # Tiny on cpu/pi, too big for the MCU.
    metrics = {"accuracy": 0.9, "latency_ms": 10.0, "size_mb": 5.0, "ram_mb": 100.0}
    rows = compare_all_profiles(metrics)
    # microcontroller_sim should be infeasible and land last.
    assert rows[0]["feasible"] is True
    assert rows[-1]["profile"] == "microcontroller_sim"
    assert rows[-1]["feasible"] is False


def test_accepts_demo_report_shape():
    demo = {
        "compressed": {
            "latency_ms": {"mean": 12.0},
            "size_mb": 4.0,
            "peak_ram_mib": 90.0,
            "synthetic_accuracy": 0.1,
        }
    }
    rows = compare_all_profiles(demo)
    assert any(r["feasible"] for r in rows)


def test_markdown_has_all_rows():
    metrics = {"accuracy": 0.9, "latency_ms": 10.0, "size_mb": 5.0, "ram_mb": 100.0}
    md = to_markdown(compare_all_profiles(metrics))
    assert "Hardware profile comparison" in md
    for name in ("cpu", "raspberry_pi", "smartphone", "microcontroller_sim"):
        assert name in md


def test_cli_writes_markdown(tmp_path):
    metrics = tmp_path / "m.json"
    metrics.write_text(json.dumps({"accuracy": 0.9, "latency_ms": 10, "size_mb": 5, "ram_mb": 100}))
    out = tmp_path / "cmp.md"
    assert main(["--metrics", str(metrics), "--out", str(out)]) == 0
    assert "Hardware profile comparison" in out.read_text()
