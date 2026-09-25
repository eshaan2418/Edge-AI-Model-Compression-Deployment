from importlib import import_module


def test_imports():
    import_module("edge_ai_compression.core.pipeline")
    import_module("edge_ai_compression.core.runner")
    import_module("edge_ai_compression.optimization.search.bayesian_search")
    import_module("legacy.model_compression.datasets")


ENTRY_MODULES = [
    "edge_ai_compression.pretraining.trainer",
    "edge_ai_compression.pretraining.signals",
    "edge_ai_compression.compression.pruning.recovery",
    "edge_ai_compression.compression.quantization.ptq",
    "edge_ai_compression.core.runner",
    "edge_ai_compression.inference.engine",
    "edge_ai_compression.experiments.run_track",
    "edge_ai_compression.experiment_db.merge",
]


def test_entry_modules_import_first_in_a_fresh_interpreter():
    # Circular imports only show up for some import orders; import each module first.
    import subprocess
    import sys

    for mod in ENTRY_MODULES:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {mod}"], capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{mod}:\n{proc.stderr[-2000:]}"
