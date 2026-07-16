from __future__ import annotations

import importlib.util
import json

import pytest

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.export import ExportError, export_model
from edge_ai_compression.interfaces import Exporter, Pruner


def _model():
    return ModelRegistry.create("small_cnn_student")


def test_torchscript_export_roundtrip(tmp_path):
    import torch

    out = tmp_path / "model.pt"
    written = export_model(_model(), out, "torchscript", input_shape=(1, 3, 32, 32))
    assert out.is_file()
    loaded = torch.jit.load(written)
    y = loaded(torch.randn(1, 3, 32, 32))
    assert y.shape[0] == 1


def test_json_export_is_metadata_only(tmp_path):
    out = tmp_path / "model.json"
    export_model(_model(), out, "json")
    data = json.loads(out.read_text())
    assert data["export_kind"] == "metadata_only"
    assert data["num_parameters"] > 0
    assert "does NOT" in data["note"]


def test_unknown_format_raises():
    with pytest.raises(ExportError, match="Unknown export format"):
        export_model(_model(), "/tmp/x.bin", "coreml")


def test_tflite_export_refuses_honestly(tmp_path):
    with pytest.raises(ExportError, match="not supported"):
        export_model(_model(), tmp_path / "m.tflite", "tflite")


def test_onnx_export_or_clear_error(tmp_path):
    out = tmp_path / "model.onnx"
    if importlib.util.find_spec("onnx") is None:
        # Dependency absent -> must raise a clear, actionable error, not crash.
        with pytest.raises(ExportError, match="onnx"):
            export_model(_model(), out, "onnx")
    else:
        export_model(_model(), out, "onnx")
        assert out.is_file()


def test_concrete_stages_satisfy_protocols():
    from edge_ai_compression.core.experiment import PruningSection
    from edge_ai_compression.core.pipeline import PruningStage

    stage = PruningStage(PruningSection(enabled=True, amount=0.5))
    assert isinstance(stage, Pruner)
    # Exporter protocol is structural; a plain object with export() satisfies it.

    class _Exp:
        def export(self, model, out_path, **kwargs):
            return out_path

    assert isinstance(_Exp(), Exporter)
