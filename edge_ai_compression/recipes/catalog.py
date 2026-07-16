"""The catalog of named compression recipes.

A recipe is a *declarative* bundle of compression settings targeting a
deployment scenario. Recipes only describe configuration — applying one merges
its settings into a base experiment config; it never trains anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    when_to_use: str
    pruning_amount: float
    pruning_mode: str
    quantization: bool
    quantization_mode: str
    distillation: bool
    hardware_profile: str
    export_format: str

    def to_config_overlay(self) -> dict[str, Any]:
        """Return the config fragment this recipe merges into a base config.

        Only touches known experiment-config keys (``compression``,
        ``hardware_profile``) so the generated config stays schema-valid.
        """
        return {
            "compression": {
                "pruning": {
                    "enabled": self.pruning_amount > 0.0,
                    "amount": self.pruning_amount,
                    "mode": self.pruning_mode,
                },
                "quantization": {
                    "enabled": self.quantization,
                    "mode": self.quantization_mode,
                },
                "distillation": {"enabled": self.distillation},
            },
            "hardware_profile": self.hardware_profile,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "pruning_amount": self.pruning_amount,
            "pruning_mode": self.pruning_mode,
            "quantization": self.quantization,
            "quantization_mode": self.quantization_mode,
            "distillation": self.distillation,
            "hardware_profile": self.hardware_profile,
            "export_format": self.export_format,
        }


_RECIPES: dict[str, Recipe] = {
    "tiny_cpu": Recipe(
        name="tiny_cpu",
        description="Light pruning only; a safe first step for laptop/server CPU.",
        when_to_use="You want a small, low-risk compression pass with minimal accuracy impact.",
        pruning_amount=0.3,
        pruning_mode="global_unstructured",
        quantization=False,
        quantization_mode="dynamic_linear",
        distillation=False,
        hardware_profile="cpu",
        export_format="torchscript",
    ),
    "raspberry_pi_fast": Recipe(
        name="raspberry_pi_fast",
        description="Moderate pruning + dynamic quantization for ARM SBCs.",
        when_to_use="Targeting a Raspberry Pi-class device where size and latency both matter.",
        pruning_amount=0.5,
        pruning_mode="global_unstructured",
        quantization=True,
        quantization_mode="dynamic_linear",
        distillation=False,
        hardware_profile="raspberry_pi",
        export_format="onnx",
    ),
    "size_first": Recipe(
        name="size_first",
        description="Aggressive pruning + quantization to minimize model size.",
        when_to_use="Storage/bandwidth is the binding constraint and some accuracy loss is OK.",
        pruning_amount=0.8,
        pruning_mode="global_unstructured",
        quantization=True,
        quantization_mode="dynamic_linear",
        distillation=False,
        hardware_profile="smartphone",
        export_format="tflite",
    ),
    "accuracy_first": Recipe(
        name="accuracy_first",
        description="Gentle pruning with distillation to preserve accuracy.",
        when_to_use="Accuracy is paramount; you accept a larger model and longer pipeline.",
        pruning_amount=0.2,
        pruning_mode="layerwise_adaptive",
        quantization=False,
        quantization_mode="dynamic_linear",
        distillation=True,
        hardware_profile="cpu",
        export_format="torchscript",
    ),
    "demo_prune_quantize": Recipe(
        name="demo_prune_quantize",
        description="The prune+quantize combo used by the offline demo.",
        when_to_use="You want to reproduce the demo's compression settings in a real experiment.",
        pruning_amount=0.5,
        pruning_mode="global_unstructured",
        quantization=True,
        quantization_mode="dynamic_linear",
        distillation=False,
        hardware_profile="cpu",
        export_format="torchscript",
    ),
}


def available_recipes() -> list[str]:
    """Return the sorted list of recipe names."""
    return sorted(_RECIPES)


def get_recipe(name: str) -> Recipe:
    """Look up a recipe by name, with a helpful error listing valid names."""
    if name not in _RECIPES:
        raise KeyError(f"Unknown recipe '{name}'. Available: {available_recipes()}")
    return _RECIPES[name]
