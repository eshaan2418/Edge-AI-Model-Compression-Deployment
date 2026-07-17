"""Hardware deployment profiles and their resource budgets.

Each profile encodes the constraints a compressed model must satisfy to be
deployable on that target. Numbers are deliberate, documented *order-of-
magnitude* budgets for planning and scoring — not measured device limits. Real
deployment should confirm them on the actual hardware.

The optional ``peak_gflops`` / ``mem_bandwidth_gbs`` fields are **nominal
vendor-spec ceilings** (theoretical peak FP32 throughput and memory bandwidth),
used only for roofline planning (see ``theory.roofline``). They are ceilings no
real workload reaches, not measurements — treat them as order-of-magnitude
planning numbers exactly like the budget fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """Resource budget for a deployment target.

    Attributes:
        name: Profile identifier.
        max_latency_ms: Upper bound on acceptable single-inference latency.
        max_size_mb: Upper bound on serialized model size.
        max_ram_mb: Upper bound on peak inference RAM.
        preferred_export_format: Recommended export target for this device.
        notes: Human-readable context for the budget choices.
        peak_gflops: Nominal peak FP32 throughput (GFLOP/s), or None if unknown.
        mem_bandwidth_gbs: Nominal peak memory bandwidth (GB/s), or None.
    """

    name: str
    max_latency_ms: float
    max_size_mb: float
    max_ram_mb: float
    preferred_export_format: str
    notes: str
    peak_gflops: float | None = None
    mem_bandwidth_gbs: float | None = None


# Documented planning budgets. See module docstring re: these being budgets,
# not measured device ceilings.
PROFILES: dict[str, HardwareProfile] = {
    "cpu": HardwareProfile(
        name="cpu",
        max_latency_ms=100.0,
        max_size_mb=500.0,
        max_ram_mb=4096.0,
        preferred_export_format="torchscript",
        notes="Laptop/server class x86 CPU. Generous budget; useful as a baseline.",
        peak_gflops=150.0,
        mem_bandwidth_gbs=40.0,
    ),
    "raspberry_pi": HardwareProfile(
        name="raspberry_pi",
        max_latency_ms=200.0,
        max_size_mb=100.0,
        max_ram_mb=512.0,
        preferred_export_format="onnx",
        notes="ARM Cortex-A class SBC (e.g. Pi 4). ONNX Runtime / TFLite recommended.",
        peak_gflops=10.0,
        mem_bandwidth_gbs=4.0,
    ),
    "smartphone": HardwareProfile(
        name="smartphone",
        max_latency_ms=50.0,
        max_size_mb=50.0,
        max_ram_mb=1024.0,
        preferred_export_format="tflite",
        notes="Mobile SoC with NNAPI/CoreML delegates; tight latency for interactivity.",
        peak_gflops=50.0,
        mem_bandwidth_gbs=25.0,
    ),
    "microcontroller_sim": HardwareProfile(
        name="microcontroller_sim",
        max_latency_ms=500.0,
        max_size_mb=1.0,
        max_ram_mb=0.5,
        preferred_export_format="tflite_micro",
        notes="Simulated MCU (e.g. Cortex-M). KB-scale RAM; only tiny models fit.",
        peak_gflops=0.1,
        mem_bandwidth_gbs=0.5,
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Look up a profile by name, with a helpful error listing valid names."""
    key = name.lower().replace("-", "_")
    if key not in PROFILES:
        raise KeyError(f"Unknown hardware profile '{name}'. Available: {sorted(PROFILES)}")
    return PROFILES[key]


def available_profiles() -> list[str]:
    return sorted(PROFILES)
