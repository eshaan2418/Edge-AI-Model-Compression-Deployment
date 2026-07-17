"""Capture a reproducibility snapshot of the runtime environment.

Everything here is best-effort and defensive: a missing package, a repo with no
git history, or a machine without torch must never crash experiment logging.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

# Packages worth pinning in a reproducibility snapshot. Absent ones are skipped.
_TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "tf-keras",
    "tensorflow-model-optimization",
    "onnx",
    "onnxruntime",
    "matplotlib",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_commit() -> str | None:
    """Return the short git commit hash, or ``None`` if git/history is absent."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def _device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"backend": "cpu", "cuda_available": False}
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["backend"] = "cuda"
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - torch optional / probing must never crash
        pass
    return info


def capture_environment() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the current environment.

    Keys: ``timestamp``, ``python``, ``platform``, ``packages``, ``git_commit``
    (nullable), and ``device``. Safe to call anywhere; never raises.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "device": _device_info(),
    }
