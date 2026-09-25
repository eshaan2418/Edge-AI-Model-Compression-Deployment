"""Quantized layer wrappers, BatchNorm folding, and model-wide layer swapping.

A ``QuantConv2d`` / ``QuantLinear`` fake-quantizes its weight (round-to-nearest,
or AdaRound's learned rounding) and, once calibrated, its *input* activation
with a static symmetric per-tensor scale. That matches where the C++ engine
quantizes (the input of each conv / linear), so a simulated model and its
engine lowering compute the same integer arithmetic.
"""

from __future__ import annotations

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from edge_ai_compression.compression.quantization.quantizer import (
    WeightSpec,
    fake_quant,
    fake_quant_weight,
    integer_weight,
    qmax,
    scale_2d,
    weight_scale,
)

# AdaRound rectified-sigmoid stretch (Nagel et al. 2020)
ZETA, GAMMA = 1.1, -0.1


class QuantLayer(nn.Module):
    """Shared weight / input quantization state for conv and linear wrappers."""

    weight: nn.Parameter
    bias: nn.Parameter | None

    def _init_quant(self, spec: WeightSpec, act_bits: int | None) -> None:
        self.spec = spec
        self.act_bits = act_bits
        self.register_buffer("w_scale", weight_scale(self.weight, spec))
        self.register_buffer("a_scale", torch.tensor(1.0))
        self.act_enabled = False  # set by calibration
        self.alpha: nn.Parameter | None = None  # AdaRound rounding logits over [M, K]
        self.hard_round = False
        self.lsq = False  # learnable step sizes (QAT)

    def enable_lsq(self) -> None:
        """Make weight and activation scales learnable with LSQ gradient scaling."""
        if self.alpha is not None:
            raise ValueError("LSQ and AdaRound rounding are mutually exclusive")
        for name in ("w_scale", "a_scale"):
            value = getattr(self, name).detach().clone()
            del self._buffers[name]
            setattr(self, name, nn.Parameter(value))
        self.lsq = True

    def _lsq_scale(self, scale: torch.Tensor, numel: int, bits: int) -> torch.Tensor:
        if not self.lsq:
            return scale
        # LSQ (Esser et al. 2020): scale the step-size gradient by 1/sqrt(numel * qmax).
        g = 1.0 / (numel * qmax(bits)) ** 0.5
        s = scale.clamp_min(1e-8)
        return (s - s * g).detach() + s * g

    # --------------------------------------------------------------- weight --
    def _w2d(self) -> torch.Tensor:
        return self.weight.reshape(self.weight.shape[0], -1)

    def _s2d(self) -> torch.Tensor:
        m, k = self._w2d().shape
        return scale_2d(self.w_scale, self.spec, m, k)

    def init_adaround(self) -> None:
        """Start learned rounding at the current fractional parts (h(alpha) = frac)."""
        w, s = self._w2d().detach(), self._s2d().detach()
        frac = w / s - torch.floor(w / s)
        p = ((frac - GAMMA) / (ZETA - GAMMA)).clamp(1e-4, 1 - 1e-4)
        self.alpha = nn.Parameter(torch.log(p / (1 - p)))
        self.hard_round = False

    def rounding(self) -> torch.Tensor:
        """AdaRound h(alpha) in [0, 1] (soft) or {0, 1} (hard)."""
        assert self.alpha is not None
        if self.hard_round:
            return (self.alpha >= 0).to(self.alpha.dtype)
        return torch.clamp(torch.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)

    def quant_weight(self) -> torch.Tensor:
        if self.alpha is None:
            per_scale = self.weight.numel() // self.w_scale.numel()
            scale = self._lsq_scale(self.w_scale, per_scale, self.spec.bits)
            return fake_quant_weight(self.weight, scale, self.spec)
        w, s, q = self._w2d(), self._s2d(), qmax(self.spec.bits)
        codes = torch.clamp(torch.floor(w.detach() / s) + self.rounding(), -q, q)
        return (codes * s).reshape(self.weight.shape)

    def integer_weight(self) -> torch.Tensor:
        """Integer codes [M, K] (int32) of the quantized weight."""
        if self.alpha is None:
            return integer_weight(self.weight, self.w_scale, self.spec)
        w, s, q = self._w2d().detach(), self._s2d().detach(), qmax(self.spec.bits)
        codes = torch.floor(w / s) + (self.alpha.detach() >= 0).to(w.dtype)
        return torch.clamp(codes, -q, q).to(torch.int32)

    # ---------------------------------------------------------------- input --
    def quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_bits is None or not self.act_enabled:
            return x
        scale = self._lsq_scale(self.a_scale, x[0].numel(), self.act_bits)
        return fake_quant(x, scale, self.act_bits)

    def extra_repr(self) -> str:
        act = f"a{self.act_bits}" if self.act_bits else "a-fp"
        rnd = "adaround" if self.alpha is not None else ("lsq" if self.lsq else "nearest")
        return f"w{self.spec.bits}/{self.spec.granularity}, {act}, {rnd}"


class QuantConv2d(QuantLayer):
    def __init__(self, conv: nn.Conv2d, spec: WeightSpec, act_bits: int | None) -> None:
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.kernel_size = conv.kernel_size
        self._init_quant(spec, act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            self.quant_input(x),
            self.quant_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantLinear(QuantLayer):
    def __init__(self, linear: nn.Linear, spec: WeightSpec, act_bits: int | None) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
        self._init_quant(spec, act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.quant_input(x), self.quant_weight(), self.bias)


def quant_layers(model: nn.Module) -> list[tuple[str, QuantLayer]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, QuantLayer)]


def _set_module(model: nn.Module, path: str, new: nn.Module) -> None:
    parent_path, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, attr, new)


def fold_bn(model: nn.Module) -> nn.Module:
    """Fold every BatchNorm2d that directly follows a Conv2d into it (in place).

    The conv gains a bias; the BN module is replaced by nn.Identity so the
    model keeps its class and forward code. Requires eval-mode statistics.
    """
    model.eval()
    gm = fx.symbolic_trace(model)
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        conv = gm.get_submodule(str(node.target))
        users = list(node.users)
        if not isinstance(conv, nn.Conv2d) or len(users) != 1:
            continue
        bn_node = users[0]
        if bn_node.op != "call_module":
            continue
        bn = gm.get_submodule(str(bn_node.target))
        if not isinstance(bn, nn.BatchNorm2d):
            continue
        real_conv = model.get_submodule(str(node.target))
        with torch.no_grad():
            inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            real_conv.weight.mul_(inv.reshape(-1, 1, 1, 1))
            b = real_conv.bias if real_conv.bias is not None else torch.zeros_like(bn.running_mean)
            new_bias = (b - bn.running_mean) * inv + bn.bias
            real_conv.bias = nn.Parameter(new_bias.detach().clone())
        _set_module(model, str(bn_node.target), nn.Identity())
    return model


def quantize_model(
    model: nn.Module,
    spec: WeightSpec,
    act_bits: int | None,
    *,
    first_last_bits: int | None = 8,
) -> list[str]:
    """Replace every Conv2d / Linear with its quantized wrapper (in place).

    With ``first_last_bits`` the first and last quantized layers (in forward
    order) use that many weight bits, per channel, as is standard practice
    (AdaRound and BRECQ keep them at 8 bits). Returns layer names in forward order.
    """
    gm = fx.symbolic_trace(model)
    order = [
        str(n.target)
        for n in gm.graph.nodes
        if n.op == "call_module"
        and isinstance(gm.get_submodule(str(n.target)), nn.Conv2d | nn.Linear)
    ]
    order = list(dict.fromkeys(order))
    for i, name in enumerate(order):
        layer = model.get_submodule(name)
        layer_spec = spec
        if first_last_bits is not None and i in (0, len(order) - 1):
            layer_spec = WeightSpec(bits=first_last_bits, granularity="per_channel")
        if isinstance(layer, nn.Conv2d):
            new: QuantLayer = QuantConv2d(layer, layer_spec, act_bits)
        else:
            new = QuantLinear(layer, layer_spec, act_bits)
        _set_module(model, name, new)
    return order
