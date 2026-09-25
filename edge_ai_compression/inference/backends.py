"""Inference backends: how a model is serialized (parent) and run (benchmark worker).

``export`` writes one artifact file; ``load`` turns it into a callable
``[B, C, H, W] float32 -> [B, classes] float32``. Heavy imports (onnxruntime,
the kernel engine) happen inside ``load`` so each backend's cold start only pays
for what it uses.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

from edge_ai_compression.benchmarking.config import BACKENDS

Runner = Callable[[np.ndarray], np.ndarray]


class Backend(Protocol):
    name: str

    def export(self, model: nn.Module, input_shape: tuple[int, ...], directory: Path) -> Path: ...

    def load(self, path: Path, num_threads: int) -> Runner: ...


class TorchEager:
    name = "torch_eager"

    def export(self, model: nn.Module, input_shape: tuple[int, ...], directory: Path) -> Path:
        path = directory / "model.pt"
        torch.save(copy.deepcopy(model).cpu().eval(), path)
        return path

    def load(self, path: Path, num_threads: int) -> Runner:
        torch.set_num_threads(num_threads)
        model = torch.load(path, map_location="cpu", weights_only=False).eval()

        def run(x: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return model(torch.from_numpy(x)).numpy()

        return run


class OnnxRuntime:
    """PyTorch -> ONNX (dynamo exporter) -> ONNX Runtime CPU, all graph optimizations on."""

    name = "onnxruntime"

    def export(self, model: nn.Module, input_shape: tuple[int, ...], directory: Path) -> Path:
        path = directory / "model.onnx"
        torch.onnx.export(
            copy.deepcopy(model).cpu().eval(),
            (torch.zeros(input_shape),),
            str(path),
            dynamo=True,
            external_data=False,  # one self-contained file, so size_mb counts the weights
            verbose=False,
            input_names=["input"],
            output_names=["logits"],
        )
        return path

    def load(self, path: Path, num_threads: int) -> Runner:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def run(x: np.ndarray) -> np.ndarray:
            return sess.run(None, {name: x})[0]

        return run


class EdgeEngine:
    """The C++ kernel engine (single-threaded, DECISIONS D2.2) in one of its modes."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.name = f"edge_{mode}"

    def export(self, model: nn.Module, input_shape: tuple[int, ...], directory: Path) -> Path:
        from edge_ai_compression.inference.engine import compile_model

        path = directory / "model.engine"
        with open(path, "wb") as f:
            pickle.dump(compile_model(copy.deepcopy(model).cpu(), self.mode), f)
        return path

    def load(self, path: Path, num_threads: int) -> Runner:
        from edge_ai_compression.inference import engine  # noqa: F401  (unpickling needs it)

        torch.set_num_threads(num_threads)  # only affects the few torch ops (maxpool)
        with open(path, "rb") as f:
            return pickle.load(f).prepare()


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend '{name}'; expected one of {BACKENDS}")
    if name == "torch_eager":
        return TorchEager()
    if name == "onnxruntime":
        return OnnxRuntime()
    return EdgeEngine(name.removeprefix("edge_"))


def run_batched(run: Runner, x: np.ndarray, batch: int) -> np.ndarray:
    """Run ``x`` through a runner whose artifact has a fixed batch size (zero-pads the tail)."""
    outs = []
    for i in range(0, len(x), batch):
        chunk = x[i : i + batch]
        n = len(chunk)
        if n < batch:
            chunk = np.concatenate([chunk, np.zeros((batch - n, *chunk.shape[1:]), chunk.dtype)])
        outs.append(run(np.ascontiguousarray(chunk, dtype=np.float32))[:n])
    return np.concatenate(outs)
