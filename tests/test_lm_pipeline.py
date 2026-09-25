from __future__ import annotations

import csv

import pytest

from edge_ai_compression.lm.pipeline import PipelineConfig, run_pipeline
from edge_ai_compression.utils.config_loader import load_yaml


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_smoke_pipeline_logs_every_stage_method_and_resumes(tmp_path):
    raw = load_yaml("edge_ai_compression/configs/lm/smoke.yml")
    raw.update(results_dir=str(tmp_path / "r"), models_dir=str(tmp_path / "m"))
    cfg = PipelineConfig.from_dict(raw)
    run_pipeline(cfg)
    runs = _rows(tmp_path / "r" / "lm_runs.csv")
    assert [r["stage"] for r in runs] == ["pretrain", "sft", "dpo", "distill"]
    evals = _rows(tmp_path / "r" / "lm_evals.csv")
    grid = {(r["stage"], r["method"]) for r in evals}
    assert grid == {(s, m) for s in ("pretrain", "sft", "dpo") for m in cfg.methods} | {
        ("pretrain", "speculative")
    }
    behavior = [r for r in evals if r["stage"] != "pretrain" and r["method"] != "speculative"]
    assert all(0.0 <= float(r["constraint_rate"]) <= 1.0 for r in behavior)
    spec = next(r for r in evals if r["method"] == "speculative")
    assert 0.0 <= float(spec["acceptance_rate"]) <= 1.0
    run_pipeline(cfg)  # resume: nothing new
    assert len(_rows(tmp_path / "r" / "lm_evals.csv")) == len(evals)
    assert len(_rows(tmp_path / "r" / "lm_runs.csv")) == 4


def test_pipeline_config_validation():
    with pytest.raises(ValueError, match="unknown ladder methods"):
        PipelineConfig.from_dict({"methods": ["w2a2"]})
    with pytest.raises(ValueError, match="unknown lm pipeline keys"):
        PipelineConfig.from_dict({"epochs": 1})
