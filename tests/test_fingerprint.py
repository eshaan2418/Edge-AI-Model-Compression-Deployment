from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest

from edge_ai_compression.benchmarking.fingerprint import check_environment, collect


@pytest.fixture(scope="module")
def fp():
    return collect()


def test_static_fields_present(fp):
    s = fp.static
    for key in ("os", "machine", "cpu", "logical_cores", "memory_bytes", "python", "torch"):
        assert s[key] is not None, key
    assert s["cpu"]["model"]
    assert isinstance(s["cpu"]["isa_features"], list)
    assert s["torch"]["version"]
    assert s["packages"]["numpy"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_perf_levels(fp):
    levels = fp.static["cpu"]["perf_levels"]
    assert levels and all(lvl["physical_cores"] > 0 for lvl in levels)


def test_json_serializable_and_hash_stable(fp):
    json.dumps(fp.to_dict(), default=str)
    assert len(fp.hash) == 16
    assert fp.hash == collect().hash


def test_hash_ignores_dynamic_state(fp):
    changed = replace(fp, dynamic={**fp.dynamic, "load_avg": [99.0, 99.0, 99.0]})
    assert changed.hash == fp.hash
    other = replace(fp, static={**fp.static, "python": "0.0.0"})
    assert other.hash != fp.hash


def test_git_commit_recorded(fp):
    assert fp.git_commit is None or len(fp.git_commit) == 40


def test_check_environment_strict(fp):
    on_battery = replace(fp, dynamic={**fp.dynamic, "power_source": "battery"})
    assert check_environment(on_battery, strict=False) == ["running on battery power"]
    with pytest.raises(RuntimeError, match="battery"):
        check_environment(on_battery, strict=True)
    clean = replace(fp, dynamic={**fp.dynamic, "power_source": "ac", "low_power_mode": False})
    assert check_environment(clean, strict=True) == []
