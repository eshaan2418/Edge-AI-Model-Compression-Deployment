"""Hardware / software fingerprint recorded with every benchmark run.

``static`` describes the machine and software stack and is hashed into
``fingerprint_hash`` so runs can be grouped by environment. ``dynamic`` is
state that can change between runs on the same machine (power source, thermal
state, load) and is recorded but not hashed. The git commit is kept separate
so runs on the same machine group together across commits.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "KMP_AFFINITY",
    "OMP_PROC_BIND",
)
LINUX_ISA_PREFIXES = ("avx", "fma", "f16c", "amx", "sse4", "asimd", "sve", "i8mm", "bf16")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def _cpuinfo() -> dict[str, str]:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return {}
    info: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() not in info:
            info[key.strip()] = value.strip()
    return info


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _torch_info() -> dict[str, Any]:
    import torch

    build = torch.__config__.show()
    blas = re.search(r"BLAS_INFO=([^,\s]+)", build)
    backend = re.search(r"ATen parallel backend: (.+)", torch.__config__.parallel_info())
    return {
        "version": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "blas": blas.group(1) if blas else None,
        "parallel_backend": backend.group(1).strip() if backend else None,
        "build_config": build,
    }


def _macos_cpu() -> dict[str, Any]:
    levels = []
    for i in range(int(_sysctl("hw.nperflevels") or 0)):
        levels.append(
            {
                "name": _sysctl(f"hw.perflevel{i}.name"),
                "physical_cores": int(_sysctl(f"hw.perflevel{i}.physicalcpu") or 0),
            }
        )
    features = []
    for line in (_run(["sysctl", "hw.optional.arm"]) or "").splitlines():
        key, _, value = line.partition(":")
        if value.strip() == "1":
            features.append(key.strip().removeprefix("hw.optional.arm."))
    return {
        "model": _sysctl("machdep.cpu.brand_string"),
        "hardware_model": _sysctl("hw.model"),
        "perf_levels": levels,
        "isa_features": sorted(features),
        "virtualized": _sysctl("kern.hv_vmm_present") == "1",
        "os_version": _sysctl("kern.osproductversion"),
    }


def _linux_cpu() -> dict[str, Any]:
    info = _cpuinfo()
    flags = (info.get("flags") or info.get("Features") or "").split()
    os_release = Path("/etc/os-release")
    pretty = None
    if os_release.is_file():
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_release.read_text(), re.MULTILINE)
        pretty = m.group(1) if m else None
    return {
        "model": info.get("model name") or info.get("CPU part"),
        "hardware_model": None,
        "perf_levels": [],
        "isa_features": sorted(f for f in flags if f.startswith(LINUX_ISA_PREFIXES)),
        "virtualized": "hypervisor" in flags,
        "os_version": pretty,
    }


def _git() -> tuple[str | None, bool | None]:
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return None, None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return commit, bool(status)


def _power_state() -> dict[str, Any]:
    power_source, low_power, thermal = "unknown", None, None
    if sys.platform == "darwin":
        batt = _run(["pmset", "-g", "batt"]) or ""
        if "AC Power" in batt:
            power_source = "ac"
        elif "Battery Power" in batt:
            power_source = "battery"
        settings = _run(["pmset", "-g"]) or ""
        m = re.search(r"^\s*(?:lowpowermode|powermode)\s+(\d+)", settings, re.MULTILINE)
        low_power = m is not None and m.group(1) == "1"
        thermal = _run(["pmset", "-g", "therm"])
    else:
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if battery is not None:
            power_source = "ac" if battery.power_plugged else "battery"
    return {"power_source": power_source, "low_power_mode": low_power, "thermal": thermal}


@dataclass(frozen=True)
class Fingerprint:
    static: dict[str, Any]
    dynamic: dict[str, Any]
    git_commit: str | None
    git_dirty: bool | None

    @property
    def hash(self) -> str:
        blob = json.dumps(self.static, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "hash": self.hash}

    def environment_problems(self) -> list[str]:
        """Conditions that make latency numbers unrepresentative on a laptop."""
        problems = []
        if self.dynamic["power_source"] == "battery":
            problems.append("running on battery power")
        if self.dynamic["low_power_mode"]:
            problems.append("Low Power Mode is enabled")
        return problems


def collect() -> Fingerprint:
    cpu = _macos_cpu() if sys.platform == "darwin" else _linux_cpu()
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    static = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {name: _package_version(name) for name in ("numpy", "scipy", "torch")},
        "torch": _torch_info(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
    }
    dynamic = {
        **_power_state(),
        "load_avg": list(os.getloadavg()),
        "cpu_governor": governor.read_text().strip() if governor.is_file() else None,
    }
    commit, dirty = _git()
    return Fingerprint(static=static, dynamic=dynamic, git_commit=commit, git_dirty=dirty)


def check_environment(fp: Fingerprint, *, strict: bool) -> list[str]:
    """Return environment problems; raise if ``strict`` and there are any."""
    problems = fp.environment_problems()
    if strict and problems:
        raise RuntimeError(
            "Refusing to benchmark: "
            + "; ".join(problems)
            + ". Fix the environment or set benchmark.strict_environment: false."
        )
    return problems
