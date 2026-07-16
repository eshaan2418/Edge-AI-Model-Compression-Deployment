from __future__ import annotations

from edge_ai_compression.analysis.search_space import (
    main,
    summarize_search_space,
    to_markdown,
)


def test_explicit_variants_counts_variants():
    config = {
        "sweep_id": "demo",
        "base_config": "base.yml",
        "variants": [
            {"order": "prune>quantize"},
            {"order": "quantize>prune"},
            {"order": "distill>prune"},
        ],
    }
    summary = summarize_search_space(config)
    assert summary["mode"] == "explicit_variants"
    assert summary["num_candidates"] == 3
    assert summary["base_config"] == "base.yml"
    order = next(d for d in summary["dimensions"] if d["name"] == "order")
    assert order["count"] == 3


def test_grid_counts_cartesian_product():
    config = {
        "pruning_amount": [0.2, 0.5, 0.8],
        "quantization": [True, False],
    }
    summary = summarize_search_space(config)
    assert summary["mode"] == "grid"
    assert summary["num_candidates"] == 6
    assert summary["has_continuous"] is False


def test_continuous_distribution_is_unbounded():
    config = {"prune_amount": {"type": "uniform", "low": 0.1, "high": 0.8}}
    summary = summarize_search_space(config)
    assert summary["has_continuous"] is True
    assert summary["num_candidates"] is None
    dim = summary["dimensions"][0]
    assert dim["kind"] == "continuous"
    assert dim["range"] == {"low": 0.1, "high": 0.8}


def test_choice_distribution_is_enumerable():
    config = {"scorer": {"type": "choice", "values": ["magnitude", "gradient"]}}
    summary = summarize_search_space(config)
    assert summary["num_candidates"] == 2
    assert summary["dimensions"][0]["kind"] == "enumerated"


def test_markdown_warns_on_large_space():
    config = {"a": list(range(20)), "b": list(range(6))}  # 120 combinations
    summary = summarize_search_space(config)
    assert summary["num_candidates"] == 120
    md = to_markdown(summary)
    assert "Large space" in md
    assert "120" in md


def test_cli_writes_markdown(tmp_path):
    cfg = tmp_path / "sweep.yml"
    cfg.write_text(
        "variants:\n  - order: a\n  - order: b\n",
        encoding="utf-8",
    )
    out = tmp_path / "search_space.md"
    assert main(["--config", str(cfg), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Search space summary" in text
    assert "Candidate count" in text


def test_cli_missing_config_returns_2(tmp_path):
    assert main(["--config", str(tmp_path / "nope.yml")]) == 2
