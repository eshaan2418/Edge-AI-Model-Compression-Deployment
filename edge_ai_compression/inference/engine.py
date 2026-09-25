"""Run a CNN on the C++ kernels.

``compile_model`` traces an ``nn.Module`` with torch.fx, folds BatchNorm into the
preceding convolution, fuses ReLU into the preceding conv / linear / residual add,
and lowers convolutions to im2col + GEMM. The engine stores plain numpy weights
and packs them lazily, so it pickles cleanly for the benchmark worker (packing
then counts toward the ``load`` cold-start stage, like a real runtime).

Modes (applied to every conv / linear layer):
- ``f32``: fp32 GEMM
- ``int8``: per-output-channel int8 weights, dynamic per-tensor int8 activations
  (absmax per layer input), exact int32 accumulation, fp32 requantization
- ``w4``: int4 weight-only (per-group scales), fp32 activations
- ``sparse24``: 2:4 structured weights. Dense weights are magnitude-pruned to 2:4
  at compile time (lossy unless the model was trained 2:4). Layers whose K is not
  a multiple of 4 (e.g. a 3-channel 3x3 stem, K=27) stay dense f32.
- ``csr``: unstructured sparse weights (the model's existing zeros)
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from edge_ai_compression.inference import kernels, packing

MODES = ("f32", "int8", "w4", "sparse24", "csr")
W4_GROUP = 32


@dataclass
class OpRecord:
    """One executed op: shape metadata for roofline analysis plus measured time."""

    name: str
    kind: str
    m: int = 0
    n: int = 0
    k: int = 0
    seconds: float = 0.0

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


@dataclass
class _Gemm:
    """A conv (lowered to im2col + GEMM) or linear layer with a quantized/sparse weight."""

    name: str
    mode: str
    weight: np.ndarray  # [M, K] fp32 after BN folding
    bias: np.ndarray  # [M]
    relu: bool
    conv: tuple[int, int, int, int] | None  # (kh, kw, stride, pad) or None for linear
    isa: str
    packed: Any = field(default=None, repr=False)
    row_scale: np.ndarray | None = field(default=None, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "packed": None, "row_scale": None}

    @property
    def effective_mode(self) -> str:
        if self.mode == "sparse24" and self.weight.shape[1] % 4:
            return "f32"
        return self.mode

    def prepare(self) -> None:
        if self.packed is not None:
            return
        w, mode = self.weight, self.effective_mode
        if mode == "f32":
            self.packed = kernels.pack_f32(w)
        elif mode == "int8":
            q, self.row_scale = packing.quantize_rows_s8(w)
            self.packed = kernels.pack_s8(q, self.isa)
        elif mode == "w4":
            self.packed = kernels.make_w4(w, W4_GROUP)
        elif mode == "sparse24":
            self.packed = kernels.make_sparse24(*packing.prune_24(w))
        elif mode == "csr":
            self.packed = kernels.make_csr(w)
        else:
            raise ValueError(f"unknown mode {mode}")

    def matmul(self, b: np.ndarray) -> np.ndarray:
        """[K, N] activations -> [M, N] outputs (bias and ReLU applied)."""
        mode, isa = self.effective_mode, self.isa
        if mode == "int8":
            scale = kernels.absmax(b) / packing.QMAX or 1.0
            acc = kernels.gemm_s8(self.packed, kernels.quantize_s8(b, scale))
            return kernels.requantize(acc, self.row_scale, scale, self.bias, self.relu)
        fn = {
            "f32": kernels.gemm_f32,
            "w4": kernels.gemm_w4,
            "sparse24": kernels.gemm_sparse24,
            "csr": kernels.gemm_csr,
        }[mode]
        return fn(self.packed, b, self.bias, self.relu, isa)

    def run(self, x: np.ndarray, rec: OpRecord | None) -> np.ndarray:
        if self.conv is None:  # linear: x is [K] -> [M]
            out = self.matmul(x.reshape(-1, 1))[:, 0]
            if rec:
                rec.m, rec.n, rec.k = self.weight.shape[0], 1, self.weight.shape[1]
            return out
        kh, kw, stride, pad = self.conv
        _, h, w = x.shape
        ho, wo = (
            kernels.conv_out_size(h, kh, stride, pad),
            kernels.conv_out_size(w, kw, stride, pad),
        )
        if self.effective_mode == "int8":
            # Quantize once before im2col (8x less data to rearrange than fp32).
            scale = kernels.absmax(x) / packing.QMAX or 1.0
            cols = kernels.im2col(kernels.quantize_s8(x, scale), kh, kw, stride, pad)
            acc = kernels.gemm_s8(self.packed, cols)
            out = kernels.requantize(acc, self.row_scale, scale, self.bias, self.relu)
        else:
            out = self.matmul(kernels.im2col(x, kh, kw, stride, pad))
        if rec:
            rec.m, rec.n, rec.k = self.weight.shape[0], ho * wo, self.weight.shape[1]
        return out.reshape(-1, ho, wo)


@dataclass
class _Node:
    name: str
    kind: str  # gemm | add | relu | maxpool | avgpool | flatten
    inputs: list[str]
    gemm: _Gemm | None = None
    relu: bool = False
    pool: tuple[int, int, int] | None = None  # maxpool (kernel, stride, pad)


def _fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d | None) -> tuple[np.ndarray, np.ndarray]:
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0])
    if bn is not None:
        inv = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = w * inv[:, None, None, None]
        b = (b - bn.running_mean.double()) * inv + bn.bias.detach().double()
    return w.reshape(w.shape[0], -1).float().numpy(), b.float().numpy()


class Engine:
    def __init__(self, nodes: list[_Node], input_name: str, output_name: str, mode: str) -> None:
        self.nodes = nodes
        self.input_name = input_name
        self.output_name = output_name
        self.mode = mode

    def prepare(self) -> Engine:
        for node in self.nodes:
            if node.gemm is not None:
                node.gemm.prepare()
        return self

    def _run_one(self, x: np.ndarray, records: list[OpRecord] | None) -> np.ndarray:
        env: dict[str, np.ndarray] = {self.input_name: x}
        for node in self.nodes:
            rec = OpRecord(node.name, node.kind) if records is not None else None
            t0 = time.perf_counter_ns()
            args = [env[i] for i in node.inputs]
            if node.kind == "gemm":
                assert node.gemm is not None
                out = node.gemm.run(args[0], rec)
            elif node.kind == "add":
                out = args[0] + args[1]
                if node.relu:
                    np.maximum(out, 0, out=out)
            elif node.kind == "relu":
                out = np.maximum(args[0], 0)
            elif node.kind == "maxpool":
                assert node.pool is not None
                k, s, p = node.pool
                t = torch.from_numpy(np.ascontiguousarray(args[0]))[None]
                out = F.max_pool2d(t, k, s, p)[0].numpy()
            elif node.kind == "avgpool":
                out = args[0].mean(axis=(1, 2), dtype=np.float32)
            elif node.kind == "flatten":
                out = args[0].reshape(-1)
            else:
                raise RuntimeError(node.kind)
            env[node.name] = out
            if rec is not None:
                rec.seconds = (time.perf_counter_ns() - t0) / 1e9
                records.append(rec)
        return env[self.output_name]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """[B, C, H, W] float32 -> [B, classes] float32."""
        self.prepare()
        x = np.ascontiguousarray(x, dtype=np.float32)
        return np.stack([self._run_one(xi, None) for xi in x])

    def profile(self, x: np.ndarray) -> tuple[np.ndarray, list[OpRecord]]:
        """Run one sample and return its output and per-op records."""
        self.prepare()
        records: list[OpRecord] = []
        out = self._run_one(np.ascontiguousarray(x, dtype=np.float32), records)
        return out, records


def compile_model(model: nn.Module, mode: str = "f32", isa: str = "auto") -> Engine:
    """Lower a (ResNet-style) CNN to an Engine. Raises on unsupported ops."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    isa = kernels.best_isa() if isa == "auto" else isa
    gm = fx.symbolic_trace(model.eval())
    graph_nodes = list(gm.graph.nodes)
    users = {n: list(n.users) for n in graph_nodes}

    def module(n: fx.Node) -> nn.Module | None:
        return gm.get_submodule(str(n.target)) if n.op == "call_module" else None

    def is_relu(n: fx.Node) -> bool:
        return isinstance(module(n), nn.ReLU) or (
            n.op == "call_function" and n.target in (F.relu, torch.relu)
        )

    alias: dict[str, str] = {}  # fused/removed node -> node that produces its value
    fused_away: set[str] = set()
    nodes: list[_Node] = []
    input_name = output_name = ""

    def src(n: Any) -> str:
        name = n.name
        while name in alias:
            name = alias[name]
        return name

    def fuse_relu(producer: fx.Node) -> bool:
        """If ``producer``'s only user is a ReLU, absorb it."""
        us = users[producer]
        if len(us) == 1 and is_relu(us[0]):
            alias[us[0].name] = producer.name
            fused_away.add(us[0].name)
            return True
        return False

    for n in graph_nodes:
        if n.name in fused_away:
            continue
        mod = module(n)
        if n.op == "placeholder":
            input_name = n.name
        elif n.op == "output":
            output_name = src(n.args[0])
        elif isinstance(mod, nn.Conv2d):
            if mod.groups != 1 or mod.dilation != (1, 1) or mod.padding_mode != "zeros":
                raise NotImplementedError(f"{n.name}: only groups=1, dilation=1, zero padding")
            if mod.stride[0] != mod.stride[1] or mod.padding[0] != mod.padding[1]:
                raise NotImplementedError(f"{n.name}: only square stride/padding")
            bn = None
            last = n
            us = users[n]
            if len(us) == 1 and isinstance(module(us[0]), nn.BatchNorm2d):
                bn = module(us[0])
                alias[us[0].name] = n.name
                fused_away.add(us[0].name)
                last = us[0]
            w, b = _fold_bn(mod, bn)  # type: ignore[arg-type]
            relu = fuse_relu(last)
            kh, kw = mod.kernel_size
            gemm = _Gemm(n.name, mode, w, b, relu, (kh, kw, mod.stride[0], mod.padding[0]), isa)
            nodes.append(_Node(n.name, "gemm", [src(n.args[0])], gemm=gemm))
        elif isinstance(mod, nn.Linear):
            w = mod.weight.detach().float().numpy()
            b = (
                mod.bias.detach().float().numpy()
                if mod.bias is not None
                else np.zeros(w.shape[0], np.float32)
            )
            relu = fuse_relu(n)
            gemm = _Gemm(n.name, mode, np.ascontiguousarray(w), b, relu, None, isa)
            nodes.append(_Node(n.name, "gemm", [src(n.args[0])], gemm=gemm))
        elif isinstance(mod, nn.BatchNorm2d):
            raise NotImplementedError(f"{n.name}: BatchNorm not directly after a conv")
        elif isinstance(mod, (nn.Identity, nn.Dropout)):
            alias[n.name] = src(n.args[0])
        elif is_relu(n):
            nodes.append(_Node(n.name, "relu", [src(n.args[0])]))
        elif n.op == "call_function" and n.target in (operator.add, operator.iadd, torch.add):
            relu = fuse_relu(n)
            nodes.append(_Node(n.name, "add", [src(a) for a in n.args[:2]], relu=relu))
        elif isinstance(mod, nn.MaxPool2d):
            k, s, p = (
                int(v if isinstance(v, int) else v[0])
                for v in (mod.kernel_size, mod.stride, mod.padding)
            )
            nodes.append(_Node(n.name, "maxpool", [src(n.args[0])], pool=(k, s, p)))
        elif isinstance(mod, nn.AdaptiveAvgPool2d):
            if tuple(np.atleast_1d(mod.output_size)) not in ((1,), (1, 1)):
                raise NotImplementedError(f"{n.name}: only global average pooling")
            nodes.append(_Node(n.name, "avgpool", [src(n.args[0])]))
        elif (
            isinstance(mod, nn.Flatten)
            or (n.op == "call_function" and n.target is torch.flatten)
            or (n.op == "call_method" and n.target in ("flatten", "view", "reshape"))
        ):
            nodes.append(_Node(n.name, "flatten", [src(n.args[0])]))
        else:
            raise NotImplementedError(f"unsupported node {n.name}: {n.op} {n.target}")
    return Engine(nodes, input_name, output_name, mode)
