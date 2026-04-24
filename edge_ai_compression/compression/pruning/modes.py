from __future__ import annotations

"""Phase 12 — pruning mode registry (structured / movement / lottery hooks).

Unstructured magnitude and layerwise adaptive are implemented under
``compression.pruning``; additional modes are extension points.
"""

from enum import Enum


class PruningMode(str, Enum):
    GLOBAL_UNSTRUCTURED = "global_unstructured"
    LAYERWISE_ADAPTIVE = "layerwise_adaptive"
    STRUCTURED_CHANNELS = "structured_channels"
    MOVEMENT = "movement"
    LOTTERY_TICKET = "lottery_ticket"
