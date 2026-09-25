"""``--set key.sub=value`` command-line overrides for YAML configs."""

from __future__ import annotations

import copy
from typing import Any

import yaml


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """["a.b=1", "c=x"] -> {"a.b": 1, "c": "x"} (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with dotted keys set (intermediate dicts created)."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        node[leaf] = value
    return out
