from __future__ import annotations

import json

from edge_ai_compression.reporting.generate_report import (
    build_report_html,
    collect_inputs,
    main,
)


def _demo_json():
    return {
        "mode": "quick",
        "data": "synthetic (torchvision.FakeData, random labels)",
        "disclaimer": "synthetic_accuracy is NOT a quality metric.",
        "baseline": {
            "num_parameters": 66602,
            "size_mb": 0.264,
            "latency_ms": {"mean": 0.44, "p50": 0.4, "p95": 0.5, "p99": 0.6},
            "synthetic_accuracy": 0.1,
        },
        "compressed": {
            "num_parameters": 65952,
            "size_mb": 0.263,
            "latency_ms": {"mean": 0.34, "p50": 0.3, "p95": 0.4, "p99": 0.5},
            "weight_sparsity": 0.5,
            "peak_ram_mib": 120.0,
            "synthetic_accuracy": 0.1,
        },
        "compression": {
            "size_ratio": 1.004,
            "latency_speedup": 1.29,
            "param_reduction": 0.01,
            "sparsity_after_pruning": 0.5,
        },
    }


def test_report_from_demo(tmp_path):
    demo = tmp_path / "report.json"
    demo.write_text(json.dumps(_demo_json()))
    collected = collect_inputs(demo=demo)
    html = build_report_html(collected)
    assert "Compression demo" in html
    assert "baseline" in html
    assert "compressed" in html
    # Honesty disclaimer must always be present.
    assert "NOT real model quality" in html


def test_missing_inputs_do_not_crash(tmp_path):
    # No files at all -> still produces a valid shell with the note.
    collected = collect_inputs(demo=tmp_path / "nope.json")
    html = build_report_html(collected)
    assert "No inputs were provided" in html


def test_hardware_section_when_profile_given(tmp_path):
    demo = tmp_path / "report.json"
    demo.write_text(json.dumps(_demo_json()))
    collected = collect_inputs(demo=demo, profile="cpu")
    assert "hardware" in collected
    html = build_report_html(collected)
    assert "Hardware feasibility" in html


def test_cli_writes_html_and_summary(tmp_path):
    demo = tmp_path / "report.json"
    demo.write_text(json.dumps(_demo_json()))
    out = tmp_path / "reports" / "index.html"
    main(["--demo", str(demo), "--out", str(out)])
    assert out.is_file()
    summary = out.with_suffix(".summary.json")
    assert summary.is_file()
    data = json.loads(summary.read_text())
    assert "demo" in data["sections"]
