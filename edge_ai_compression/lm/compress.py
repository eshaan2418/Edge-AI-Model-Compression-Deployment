"""Quantization ladder for GPTs: capability (perplexity) vs behavior (constraint rate).

Block linears (qkv, proj, fc1, fc2) become QuantLinear layers from Phase 3; token
and position embeddings and the tied LM head stay in float. Activation-quantized
rungs are calibrated on pre-training windows. Accuracy-type metrics come from the
fake-quant simulation. Speed metrics are meaningful for the float model (and the
speculative-decoding row) only, since simulated quantization is not faster.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.benchmarking.fingerprint import collect
from edge_ai_compression.compression.quantization.calibration import calibrate_activations
from edge_ai_compression.compression.quantization.modules import QuantLinear
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.lm.data import InstructExample, TokenWindows
from edge_ai_compression.lm.eval import (
    constraint_rate,
    generation_speed,
    perplexity,
    speculative_generate,
)
from edge_ai_compression.lm.model import GPT
from edge_ai_compression.lm.tokenizer import BOS, encode

LM_LADDER: dict[str, dict[str, Any] | None] = {
    "fp": None,
    "w8a8": {"bits": 8, "granularity": "per_channel", "act_bits": 8},
    "w8a16": {"bits": 8, "granularity": "per_channel", "act_bits": None},
    "w4a16_g32": {"bits": 4, "granularity": "per_group", "group_size": 32, "act_bits": None},
    "w4a8": {"bits": 4, "granularity": "per_channel", "act_bits": 8},
    "w3a16_g32": {"bits": 3, "granularity": "per_group", "group_size": 32, "act_bits": None},
}
EVAL_FIELDS = (
    "eval_id",
    "timestamp",
    "schema_version",
    "git_commit",
    "fingerprint_hash",
    "source_run_id",
    "stage",
    "model",
    "seed",
    "method",
    "perplexity",
    "constraint_rate",
    "n_constraint",
    "ttft_ms",
    "decode_tokens_per_s",
    "kv_cache_bytes",
    "acceptance_rate",
    "device",
)


def quantize_gpt(model: GPT, method: str, calib_texts: list[str], device: str) -> GPT:
    """Swap block linears for QuantLinear (in place) and calibrate activations."""
    spec = LM_LADDER[method]
    if spec is None:
        return model
    ws = WeightSpec(
        bits=spec["bits"], granularity=spec["granularity"], group_size=spec.get("group_size")
    )
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name != "lm_head":
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, QuantLinear(mod, ws, spec["act_bits"]))
    if spec["act_bits"] is not None:
        loader = DataLoader(TokenWindows(calib_texts, model.cfg.block), batch_size=8)
        calibrate_activations(model, loader, num_samples=64, device=device)
    return model.eval()


def evaluate(
    model: GPT, eval_texts: list[str], examples: list[InstructExample], max_new: int
) -> dict[str, Any]:
    speed = generation_speed(model, [BOS, *encode(" Once upon a time")], new_tokens=32, runs=3)
    return {
        "perplexity": perplexity(model, eval_texts),
        "constraint_rate": constraint_rate(model, examples, max_new) if examples else None,
        "n_constraint": len(examples),
        "ttft_ms": speed.ttft_ms,
        "decode_tokens_per_s": speed.decode_tokens_per_s,
        "kv_cache_bytes": speed.kv_cache_bytes,
        "acceptance_rate": None,
    }


def speculative_row(target: GPT, draft: GPT, prompt: list[int], new_tokens: int, k: int) -> dict:
    t0 = time.perf_counter()
    res = speculative_generate(target, draft, prompt, new_tokens, k)
    seconds = time.perf_counter() - t0
    return {
        "perplexity": None,
        "constraint_rate": None,
        "n_constraint": 0,
        "ttft_ms": None,
        "decode_tokens_per_s": len(res.tokens) / max(seconds, 1e-12),
        "kv_cache_bytes": None,
        "acceptance_rate": res.acceptance_rate,
    }


def log_eval(results_dir: Path, meta: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    from edge_ai_compression.experiment_db.writer import append_row

    fp = collect()
    row = {
        "eval_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "git_commit": fp.git_commit,
        "fingerprint_hash": fp.hash,
        **meta,
        **metrics,
    }
    append_row(results_dir / "lm_evals.csv", EVAL_FIELDS, {k: row.get(k) for k in EVAL_FIELDS})
    return row
