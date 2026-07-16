from __future__ import annotations

from edge_ai_compression.experiment_db.environment import capture_environment


def test_capture_environment_shape():
    env = capture_environment()
    assert set(env) >= {"timestamp", "python", "platform", "packages", "git_commit", "device"}
    # python version looks like "3.11.5"
    assert env["python"][0].isdigit()
    assert isinstance(env["packages"], dict)
    # torch is a hard dependency, so it must be captured.
    assert "torch" in env["packages"]
    # git_commit is either a non-empty string or None (never crashes).
    assert env["git_commit"] is None or isinstance(env["git_commit"], str)
    assert env["device"]["backend"] in ("cpu", "cuda")
