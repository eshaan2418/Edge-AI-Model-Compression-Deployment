from __future__ import annotations

from edge_ai_compression.config.validate import main, validate_config


def _valid():
    return {
        "model": "resnet18_cifar",
        "dataset": "fake",
        "batch_size": 32,
        "compression": {"pruning": {"enabled": True, "amount": 0.5}},
    }


def test_valid_config_passes():
    assert validate_config(_valid()) == []


def test_smoke_config_passes():
    from pathlib import Path

    import yaml

    text = Path("edge_ai_compression/configs/experiments/smoke_cpu.yml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    errors = [i for i in validate_config(cfg) if i.severity == "error"]
    assert errors == [], errors


def test_missing_required_field_fails():
    cfg = _valid()
    del cfg["model"]
    issues = validate_config(cfg)
    assert any(i.field == "model" and i.severity == "error" for i in issues)


def test_unknown_model_fails():
    cfg = _valid()
    cfg["model"] = "not_a_real_model"
    issues = validate_config(cfg)
    assert any(i.field == "model" and "unknown" in i.message for i in issues)


def test_unknown_dataset_fails():
    cfg = _valid()
    cfg["dataset"] = "imagenet22k"
    issues = validate_config(cfg)
    assert any(i.field == "dataset" for i in issues)


def test_bad_pruning_amount_fails():
    cfg = _valid()
    cfg["compression"]["pruning"]["amount"] = 1.5
    issues = validate_config(cfg)
    assert any(i.field == "compression.pruning.amount" for i in issues)


def test_strict_mode_catches_unknown_keys():
    cfg = _valid()
    cfg["bach_size"] = 64  # typo
    assert validate_config(cfg, strict=False) == []
    issues = validate_config(cfg, strict=True)
    assert any(i.field == "bach_size" for i in issues)


def test_cli_valid_returns_zero(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("model: resnet18_cifar\ndataset: fake\n")
    assert main([str(p)]) == 0


def test_cli_invalid_returns_one(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("dataset: fake\n")  # missing model
    assert main([str(p)]) == 1
