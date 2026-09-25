"""Every YAML config in the repo parses (sweeps: every expanded variant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.core.experiment import ExperimentConfig
from edge_ai_compression.experiments.launch_sweep import expand_variants, fill_templates
from edge_ai_compression.experiments.run_kernel_study import KEYS as STUDY_KEYS
from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.utils.config_loader import load_yaml, merge_dict

ROOT = Path("edge_ai_compression/configs")


@pytest.mark.parametrize("path", sorted((ROOT / "experiments").rglob("*.yml")), ids=str)
def test_experiment_configs_parse(path):
    ExperimentConfig.from_dict(load_yaml(path))


@pytest.mark.parametrize("path", sorted((ROOT / "sweeps").glob("*.yml")), ids=str)
def test_sweep_variants_parse(path):
    spec = load_yaml(path)
    base = load_yaml(spec["base_config"])
    paths = set()
    for variant in expand_variants(spec):
        merged = fill_templates(merge_dict(base, variant))
        if spec.get("kind") == "train":
            cfg = TrainConfig.from_dict(merged)
            assert cfg.export_path not in paths, f"duplicate export_path {cfg.export_path}"
            paths.add(cfg.export_path)
        else:
            ExperimentConfig.from_dict(merged)


@pytest.mark.parametrize("path", sorted((ROOT / "train").glob("*.yml")), ids=str)
def test_train_configs_parse(path):
    TrainConfig.from_dict(load_yaml(path))


@pytest.mark.parametrize("path", sorted((ROOT / "studies").glob("*.yml")), ids=str)
def test_study_configs_parse(path):
    cfg = load_yaml(path)
    assert set(cfg) <= STUDY_KEYS
    BenchmarkConfig.from_dict(cfg.get("benchmark") or {})


@pytest.mark.parametrize("path", sorted((ROOT / "lm").glob("*.yml")), ids=str)
def test_lm_pipeline_configs_parse(path):
    from edge_ai_compression.lm.pipeline import PipelineConfig

    PipelineConfig.from_dict(load_yaml(path))
