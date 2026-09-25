from __future__ import annotations

import csv

import pytest
import torch

from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.trainer import log_spaced_steps, lr_at, run_training
from edge_ai_compression.utils.config_loader import load_yaml

SMOKE = "edge_ai_compression/configs/train/smoke_train.yml"


def test_log_spaced_steps():
    s = log_spaced_steps(1000, 5)
    assert s[0] == 1 and s[-1] == 1000 and s == sorted(set(s))
    assert log_spaced_steps(3, 10) == [1, 2, 3]


def test_lr_schedule_warmup_then_cosine():
    cfg = TrainConfig(lr=0.1, warmup_epochs=1.0, schedule="cosine")
    assert lr_at(0, 100, cfg, steps_per_epoch=10) == pytest.approx(0.01)
    assert lr_at(10, 100, cfg, steps_per_epoch=10) == pytest.approx(0.1)
    assert lr_at(99, 100, cfg, steps_per_epoch=10) < 0.001


def test_config_rejects_unknown_keys_and_variants():
    with pytest.raises(ValueError, match="unknown train keys"):
        TrainConfig.from_dict({"epochz": 1})
    with pytest.raises(ValueError, match="variant"):
        TrainConfig.from_dict({"variant": "rigl"})
    assert TrainConfig.from_dict({"limit_samples": "16"}).limit_samples == 16


def test_smoke_training_logs_run_and_checkpoints(tmp_path):
    raw = load_yaml(SMOKE)
    raw.update(checkpoint_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "results"))
    result = run_training(TrainConfig.from_dict(raw))
    assert result.steps == 8  # 2 epochs x ceil(32 / 8)
    assert len(result.checkpoints) == 3 and result.checkpoints[-1].endswith("step_0000008.pt")
    ckpt = torch.load(result.checkpoints[-1], weights_only=False)
    assert ckpt["step"] == 8 and "model_state" in ckpt
    with open(tmp_path / "results" / "training_runs.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["run_id"] == result.run_id and row["steps"] == "8"
    assert (result.run_dir / "history.jsonl").read_text().count("\n") == 4
