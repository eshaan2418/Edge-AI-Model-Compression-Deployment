"""Build a reproducibility manifest for an experiment config.

    python -m edge_ai_compression.repro.manifest \
        --config edge_ai_compression/configs/experiments/smoke_cpu.yml \
        --out reports/repro_manifest.json

Captures git state, environment, package versions, a content hash of the config,
the invoking command, timestamp, and any seeds found in the config — everything
needed to reproduce (or at least fully describe) a run. Best-effort: works even
outside a git repo or with git unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from edge_ai_compression.experiment_db.environment import capture_environment


def _git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes; None if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _find_seeds(config: Any) -> dict[str, Any]:
    """Recursively collect any key containing 'seed' from a nested config."""
    seeds: dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                if "seed" in str(k).lower() and not isinstance(v, (dict, list)):
                    seeds[path] = v
                walk(path, v)

    walk("", config)
    return seeds


def _config_hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_manifest(
    config_path: Path,
    *,
    command: str | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable reproducibility manifest for ``config_path``."""
    raw = config_path.read_bytes()
    parsed = yaml.safe_load(raw.decode("utf-8")) or {}
    env = capture_environment()

    return {
        "schema_version": 1,
        "timestamp": env.get("timestamp"),
        "command": command or ("python " + " ".join(sys.argv)),
        "git": {
            "commit": env.get("git_commit"),
            "dirty": _git_dirty(),
        },
        "python": env.get("python"),
        "python_implementation": env.get("python_implementation"),
        "platform": env.get("platform"),
        "machine": env.get("machine"),
        "device": env.get("device"),
        "packages": env.get("packages", {}),
        "config": {
            "path": str(config_path),
            "hash": _config_hash(raw),
            "seeds": _find_seeds(parsed),
        },
        "artifacts": list(artifacts or []),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.repro.manifest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Experiment YAML config.")
    p.add_argument("--out", type=Path, default=Path("reports/repro_manifest.json"))
    p.add_argument("--command", default=None, help="Override the recorded command string.")
    p.add_argument("--artifact", action="append", default=None, help="Artifact path (repeatable).")
    args = p.parse_args(argv)

    if not args.config.is_file():
        print(f"Config file not found: {args.config}")
        return 2

    manifest = build_manifest(args.config, command=args.command, artifacts=args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  git commit : {manifest['git']['commit']}  (dirty={manifest['git']['dirty']})")
    print(f"  config hash: {manifest['config']['hash']}")
    print(f"  seeds      : {manifest['config']['seeds'] or '(none found)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
