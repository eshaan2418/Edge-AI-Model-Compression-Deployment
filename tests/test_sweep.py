from __future__ import annotations

import json

import pytest
import yaml

from edge_ai_compression.experiments.launch_sweep import expand_variants, run_sweep


def test_expand_variants_list_and_axes():
    spec = {
        "variants": [{"a": 1}],
        "axes": [[{"seed": 0}, {"seed": 1}], [{"q": {"b": 4}}, {"q": {"b": 8}}]],
    }
    out = expand_variants(spec)
    assert out[0] == {"a": 1} and len(out) == 5
    assert {"seed": 1, "q": {"b": 8}} in out
    with pytest.raises(ValueError):
        expand_variants({})


def test_sweep_resumes_and_skips_done_variants(tmp_path):
    with open("edge_ai_compression/configs/experiments/smoke_benchmark.yml") as f:
        base = yaml.safe_load(f)
    base["experiment_db"]["results_dir"] = str(tmp_path / "db")
    base["checkpoint_out"] = str(tmp_path / "m.pt")
    base["benchmark"]["process_repeats"] = 1
    base_path = tmp_path / "base.yml"
    base_path.write_text(yaml.safe_dump(base))
    spec = {"sweep_id": "t", "base_config": str(base_path), "axes": [[{"seed": 0}, {"seed": 1}]]}
    state = run_sweep(spec, tmp_path / "sweeps")
    assert len(state.read_text().splitlines()) == 2
    run_sweep(spec, tmp_path / "sweeps")  # second run: everything already done
    rows = [json.loads(line) for line in state.read_text().splitlines()]
    assert len(rows) == 2 and {r["variant"]["seed"] for r in rows} == {0, 1}
