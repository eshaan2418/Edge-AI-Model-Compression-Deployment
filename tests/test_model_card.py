from __future__ import annotations

import json

from edge_ai_compression.reporting.model_card import build_model_card, main


def _synthetic_metrics():
    return {
        "compressed": {
            "latency_ms": {"mean": 12.0},
            "size_mb": 4.0,
            "peak_ram_mib": 90.0,
            "num_parameters": 66000,
            "synthetic_accuracy": 0.1,
        }
    }


def test_card_flags_synthetic_and_not_production_ready():
    card = build_model_card(_synthetic_metrics(), model_name="m", dataset="fake", profile="cpu")
    assert "SYNTHETIC" in card
    assert "Not production-ready" in card
    assert "## Limitations" in card
    assert "Ethical" in card


def test_card_includes_hardware_feasibility():
    card = build_model_card(
        _synthetic_metrics(), model_name="m", dataset="fake", profile="microcontroller_sim"
    )
    assert "microcontroller_sim" in card
    # 4 MB model exceeds the 1 MB MCU budget -> infeasible.
    assert "INFEASIBLE" in card


def test_card_handles_no_metrics():
    card = build_model_card(None, model_name="m", dataset="cifar10")
    assert "No metrics supplied" in card
    assert "Not production-ready" in card


def test_real_metrics_do_not_say_not_production_ready():
    real = {"accuracy": 0.92, "latency_ms": {"mean": 10.0}, "size_mb": 5.0}
    card = build_model_card(real, model_name="m", dataset="cifar10", profile="cpu")
    assert "Not production-ready" not in card
    assert "on-device validation" in card


def test_cli_writes_card(tmp_path):
    metrics = tmp_path / "m.json"
    metrics.write_text(json.dumps(_synthetic_metrics()))
    out = tmp_path / "card.md"
    assert main(["--metrics", str(metrics), "--profile", "cpu", "--out", str(out)]) == 0
    assert "Model Card" in out.read_text()
