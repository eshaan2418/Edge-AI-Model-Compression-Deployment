from __future__ import annotations

import gc
import os

import pytest

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.timing import apply_cpu_affinity, measure_latency


class FakeClock:
    """Advances by ``tick`` ns on every read, so each timed call lasts exactly one tick."""

    def __init__(self, tick: int = 1000) -> None:
        self.now = 0
        self.tick = tick

    def __call__(self) -> int:
        self.now += self.tick
        return self.now


def _cfg(**kw):
    base = {"warmup_iters": 0, "min_warmup_s": 0.0, "iters": 1, "min_time_s": 0.0}
    return BenchmarkConfig(**{**base, **kw})


def test_warmup_calls_are_not_recorded():
    calls = []
    trace = measure_latency(
        lambda: calls.append(1), _cfg(warmup_iters=3, iters=5), clock=FakeClock()
    )
    assert len(calls) == 8
    assert trace.tolist() == [1000] * 5
    assert trace.dtype.name == "int64"


def test_min_time_extends_iterations():
    # Each iteration consumes two ticks (2000 ns); 10_000 ns needs 5 iterations.
    calls = []
    trace = measure_latency(
        lambda: calls.append(1), _cfg(iters=1, min_time_s=1e-5), clock=FakeClock()
    )
    assert len(trace) == 5
    assert len(calls) == 5


def test_min_warmup_seconds_extends_warmup():
    calls = []
    trace = measure_latency(
        lambda: calls.append(1), _cfg(warmup_iters=1, min_warmup_s=1e-5, iters=2), clock=FakeClock()
    )
    assert len(calls) == 5 + 2
    assert len(trace) == 2


def test_gc_restored_after_exception():
    assert gc.isenabled()

    def boom() -> None:
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        measure_latency(boom, _cfg())
    assert gc.isenabled()


def test_gc_disabled_while_timing():
    seen = []
    measure_latency(lambda: seen.append(gc.isenabled()), _cfg(warmup_iters=1, iters=2))
    assert seen == [False, False, False]


def test_real_clock_positive():
    trace = measure_latency(lambda: sum(range(100)), _cfg(iters=10))
    assert (trace > 0).all()


def test_cpu_affinity_none_is_noop():
    apply_cpu_affinity(None)


@pytest.mark.skipif(hasattr(os, "sched_setaffinity"), reason="platform supports affinity")
def test_cpu_affinity_unsupported_platform_raises():
    with pytest.raises(ValueError):
        apply_cpu_affinity((0,))


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_cpu_affinity_linux_pins():
    original = os.sched_getaffinity(0)
    core = min(original)
    try:
        apply_cpu_affinity((core,))
        assert os.sched_getaffinity(0) == {core}
    finally:
        os.sched_setaffinity(0, original)


def test_config_rejects_renamed_and_unknown_keys():
    with pytest.raises(ValueError, match="renamed to 'iters'"):
        BenchmarkConfig.from_dict({"latency_repeats": 5})
    with pytest.raises(ValueError, match="removed"):
        BenchmarkConfig.from_dict({"results_md": "x.md"})
    with pytest.raises(ValueError, match="renamed to 'batch_size"):
        BenchmarkConfig.from_dict({"input_shape": [1, 3, 32, 32]})
    with pytest.raises(ValueError, match="unknown benchmark key"):
        BenchmarkConfig.from_dict({"itres": 5})


def test_config_parses_and_validates():
    cfg = BenchmarkConfig.from_dict({"batch_size": 8, "iters": "20", "cpu_affinity": [0, 1]})
    assert cfg.batch_size == 8
    assert cfg.iters == 20
    assert cfg.cpu_affinity == (0, 1)
    assert BenchmarkConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        BenchmarkConfig.from_dict({"iters": 0})
    with pytest.raises(ValueError):
        BenchmarkConfig.from_dict({"min_time_s": -1})
