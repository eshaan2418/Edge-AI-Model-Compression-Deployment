"""Benchmark one saved model in a fresh process.

Run as ``python -m edge_ai_compression.benchmarking.worker REQUEST.json RESULT.json``
by ``isolation.run_isolated``; not meant to be invoked by hand.

Cold-start stages: ``startup`` (spawn + interpreter, up to the first line of
this module), ``import`` (torch-free harness modules, which import numpy, plus
``import torch``), ``load`` (torch.load), ``first_inference``. Nothing above
``T_START_NS`` may import torch.
"""

from __future__ import annotations

import time

T_START_NS = time.monotonic_ns()

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from edge_ai_compression.benchmarking.config import BenchmarkConfig  # noqa: E402
from edge_ai_compression.benchmarking.energy import get_meter  # noqa: E402
from edge_ai_compression.benchmarking.memory import MIB, peak_rss_bytes  # noqa: E402
from edge_ai_compression.benchmarking.timing import (  # noqa: E402
    apply_cpu_affinity,
    measure_latency,
)

ENERGY_DURATION_S = 10.0


def main(argv: list[str]) -> None:
    request_path, result_path = Path(argv[0]), Path(argv[1])
    req = json.loads(request_path.read_text())
    cfg = BenchmarkConfig.from_dict(req["config"])
    apply_cpu_affinity(cfg.cpu_affinity)

    import torch

    from edge_ai_compression.utils.quant_engine import ensure_quantized_engine

    t_import = time.monotonic_ns()
    runtime_peak = peak_rss_bytes()
    torch.set_num_threads(cfg.num_threads)
    ensure_quantized_engine()

    model = torch.load(req["model_path"], map_location="cpu", weights_only=False)
    model.eval()
    t_load = time.monotonic_ns()

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(tuple(req["input_shape"]), generator=gen)

    def step() -> None:
        model(x)

    with torch.inference_mode():
        step()
        t_first = time.monotonic_ns()
        trace = measure_latency(step, cfg)
        energy = get_meter(cfg.energy_meter).joules_per_call(step, ENERGY_DURATION_S)

    peak = peak_rss_bytes()
    t_spawn = int(req["t_spawn_ns"])
    result = {
        "trace_ns": trace.tolist(),
        "stages_ms": {
            "startup": (T_START_NS - t_spawn) / 1e6,
            "import": (t_import - T_START_NS) / 1e6,
            "load": (t_load - t_import) / 1e6,
            "first_inference": (t_first - t_load) / 1e6,
            "cold_start": (t_first - t_spawn) / 1e6,
        },
        "peak_rss_mib": peak / MIB,
        "model_peak_rss_mib": (peak - runtime_peak) / MIB,
        "energy_j_per_inf": energy,
        "num_threads": torch.get_num_threads(),
    }
    result_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
