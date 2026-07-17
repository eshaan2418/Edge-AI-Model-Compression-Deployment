from __future__ import annotations

import json
import math

from edge_ai_compression.analysis.layer_sensitivity import (
    main,
    prunable_layers,
    sensitivity_profile,
)
from edge_ai_compression.core.registry import ModelRegistry


def _model():
    return ModelRegistry.create("small_cnn_student", num_classes=10)


def test_prunable_layers_found():
    layers = prunable_layers(_model())
    assert len(layers) >= 1
    assert all(hasattr(m, "weight") for _, m in layers)


def test_profile_covers_every_prunable_layer():
    model = _model()
    n_layers = len(prunable_layers(model))
    rows = sensitivity_profile(model, method="prune", amount=0.5, seed=0)
    assert len(rows) == n_layers
    for r in rows:
        assert r["drift_rel_l2"] >= 0.0
        assert math.isfinite(r["drift_rel_l2"])
        assert -1.0001 <= r["cosine"] <= 1.0001


def test_zero_prune_amount_is_sanity_zero_drift():
    rows = sensitivity_profile(_model(), method="prune", amount=0.0, seed=0)
    assert all(r["drift_rel_l2"] < 1e-6 for r in rows)


def test_positive_prune_amount_causes_drift():
    rows = sensitivity_profile(_model(), method="prune", amount=0.9, seed=0)
    assert any(r["drift_rel_l2"] > 0.0 for r in rows)


def test_profile_sorted_descending():
    rows = sensitivity_profile(_model(), method="prune", amount=0.5, seed=0)
    drifts = [r["drift_rel_l2"] for r in rows]
    assert drifts == sorted(drifts, reverse=True)


def test_quantize_method_runs():
    rows = sensitivity_profile(_model(), method="quantize", bits=4, seed=0)
    assert len(rows) >= 1
    assert all(math.isfinite(r["drift_rel_l2"]) for r in rows)


def test_deterministic():
    # Same model instance + same seed must give identical results (the profile
    # does not mutate the input model, so re-running is safe).
    model = _model()
    a = sensitivity_profile(model, method="prune", amount=0.5, seed=123)
    b = sensitivity_profile(model, method="prune", amount=0.5, seed=123)
    assert [r["drift_rel_l2"] for r in a] == [r["drift_rel_l2"] for r in b]


def test_cli_markdown(tmp_path):
    out = tmp_path / "sens.md"
    rc = main(
        ["--model", "small_cnn_student", "--method", "prune", "--amount", "0.5", "--out", str(out)]
    )
    assert rc == 0
    assert "Layer sensitivity profile" in out.read_text(encoding="utf-8")


def test_cli_json(tmp_path):
    out = tmp_path / "sens.json"
    rc = main(
        ["--model", "small_cnn_student", "--method", "quantize", "--bits", "8", "--out", str(out)]
    )
    assert rc == 0
    assert isinstance(json.loads(out.read_text(encoding="utf-8")), list)


def test_cli_unknown_model_returns_1():
    assert main(["--model", "nope"]) == 1
