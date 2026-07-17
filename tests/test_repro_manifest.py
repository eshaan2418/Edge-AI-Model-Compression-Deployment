from __future__ import annotations

import json

from edge_ai_compression.repro.manifest import build_manifest, main


def _write_config(tmp_path, seed=42):
    p = tmp_path / "c.yml"
    p.write_text(f"model: resnet18_cifar\ndataset: fake\nseed: {seed}\n")
    return p


def test_manifest_has_required_fields(tmp_path):
    m = build_manifest(_write_config(tmp_path))
    for key in ("timestamp", "command", "git", "python", "platform", "packages", "config"):
        assert key in m
    assert m["config"]["hash"].startswith("sha256:")
    assert m["config"]["seeds"] == {"seed": 42}


def test_manifest_works_without_git(tmp_path, monkeypatch):
    # Force git lookups to fail; manifest must still build.
    import edge_ai_compression.experiment_db.environment as env
    import edge_ai_compression.repro.manifest as man

    def _boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(env.subprocess, "run", _boom)
    monkeypatch.setattr(man.subprocess, "run", _boom)
    m = build_manifest(_write_config(tmp_path))
    assert m["git"]["commit"] is None
    assert m["git"]["dirty"] is None


def test_config_hash_changes_with_content(tmp_path):
    a = build_manifest(_write_config(tmp_path, seed=1))
    b = build_manifest(_write_config(tmp_path, seed=2))
    assert a["config"]["hash"] != b["config"]["hash"]


def test_cli_writes_manifest(tmp_path):
    cfg = _write_config(tmp_path)
    out = tmp_path / "manifest.json"
    assert main(["--config", str(cfg), "--out", str(out), "--artifact", "models/m.pt"]) == 0
    data = json.loads(out.read_text())
    assert data["artifacts"] == ["models/m.pt"]
