from __future__ import annotations

import yaml

from edge_ai_compression.config.validate import validate_config
from edge_ai_compression.recipes import apply as apply_cli
from edge_ai_compression.recipes import list as list_cli
from edge_ai_compression.recipes import show as show_cli
from edge_ai_compression.recipes.apply import apply_recipe, deep_merge
from edge_ai_compression.recipes.catalog import available_recipes, get_recipe


def test_catalog_has_expected_recipes():
    names = available_recipes()
    for expected in (
        "tiny_cpu",
        "raspberry_pi_fast",
        "size_first",
        "accuracy_first",
        "demo_prune_quantize",
    ):
        assert expected in names


def test_get_recipe_metadata():
    r = get_recipe("raspberry_pi_fast")
    assert r.hardware_profile == "raspberry_pi"
    assert r.quantization is True
    assert r.export_format == "onnx"


def test_get_unknown_recipe_raises():
    import pytest

    with pytest.raises(KeyError):
        get_recipe("nope")


def test_deep_merge_is_recursive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overlay = {"a": {"y": 20, "z": 30}}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    # Base is not mutated.
    assert base["a"] == {"x": 1, "y": 2}


def test_apply_recipe_produces_valid_config():
    base = {"model": "resnet18_cifar", "dataset": "fake"}
    merged = apply_recipe(base, "raspberry_pi_fast")
    assert merged["compression"]["quantization"]["enabled"] is True
    assert merged["hardware_profile"] == "raspberry_pi"
    # The generated config must still pass validation.
    errors = [i for i in validate_config(merged) if i.severity == "error"]
    assert errors == [], errors


def test_apply_cli_writes_yaml(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text("model: resnet18_cifar\ndataset: fake\n")
    out = tmp_path / "gen.yml"
    rc = apply_cli.main(["--recipe", "tiny_cpu", "--base-config", str(base), "--out", str(out)])
    assert rc == 0
    written = yaml.safe_load(out.read_text())
    assert written["compression"]["pruning"]["amount"] == 0.3


def test_list_and_show_clis_run(capsys):
    list_cli.main([])
    out = capsys.readouterr().out
    assert "raspberry_pi_fast" in out

    assert show_cli.main(["size_first"]) == 0
    out = capsys.readouterr().out
    assert "size_first" in out

    assert show_cli.main(["does_not_exist"]) == 1
