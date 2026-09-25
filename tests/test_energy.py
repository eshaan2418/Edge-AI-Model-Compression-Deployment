from __future__ import annotations

import pytest

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.energy import NullMeter, get_meter


def test_null_meter_returns_none_without_running():
    calls = []
    assert NullMeter().joules_per_call(lambda: calls.append(1), 1.0) is None
    assert calls == []


def test_get_meter_known_and_unknown():
    assert get_meter("none").name == "none"
    with pytest.raises(ValueError, match="implemented"):
        get_meter("powermetrics")


def test_config_rejects_unimplemented_meter():
    with pytest.raises(ValueError, match="energy_meter"):
        BenchmarkConfig(energy_meter="rapl")
