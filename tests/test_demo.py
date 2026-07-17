from __future__ import annotations

import json

from edge_ai_compression.demo import main, run_demo


def test_run_demo_reports_tradeoff():
    report = run_demo(quick=True)
    assert report["mode"] == "quick"
    assert "synthetic" in report["data"].lower()
    for key in ("baseline", "compressed", "compression"):
        assert key in report

    b = report["baseline"]
    c = report["compressed"]
    assert b["num_parameters"] > 0
    assert b["size_mb"] > 0
    assert c["latency_ms"]["mean"] > 0
    # Synthetic accuracy is present and clearly bounded [0, 1].
    assert 0.0 <= b["synthetic_accuracy"] <= 1.0
    assert 0.0 <= c["synthetic_accuracy"] <= 1.0
    # Pruning must have introduced sparsity.
    assert report["compression"]["sparsity_after_pruning"] > 0.0
    # Fully JSON serializable.
    json.dumps(report)


def test_demo_disclaimer_flags_synthetic():
    report = run_demo(quick=True)
    assert "NOT a quality metric" in report["disclaimer"]


def test_demo_cli_writes_reports(tmp_path):
    main(["--quick", "--out", str(tmp_path)])
    assert (tmp_path / "report.json").is_file()
    md = (tmp_path / "report.md").read_text()
    assert "synthetic" in md.lower()
    assert "Params" in md
