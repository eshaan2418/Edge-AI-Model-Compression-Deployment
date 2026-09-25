"""Capability, behavior, and inference-speed metrics for the small-LM track.

- ``perplexity``: exp(mean next-token cross-entropy) on held-out text (capability)
- ``constraint_rate``: fraction of greedy instruction completions that contain every
  required word (post-trained behavior)
- ``generation_speed``: time-to-first-token, decode tokens/s, KV-cache bytes
- ``speculative_generate``: greedy speculative decoding (Leviathan et al. 2023;
  Chen et al. 2023) with a draft model; lossless for greedy decoding (output is
  identical to the target's own greedy output), reports acceptance rate
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from edge_ai_compression.lm.data import InstructExample, satisfies
from edge_ai_compression.lm.model import GPT
from edge_ai_compression.lm.tokenizer import BOS, EOS, SEP, decode, encode


@torch.no_grad()
def perplexity(model: GPT, texts: list[str], max_tokens: int = 50_000) -> float:
    """Token-weighted perplexity over non-overlapping context-length chunks."""
    model.eval()
    device = model.tok_emb.weight.device
    block = model.cfg.block
    stream: list[int] = []
    for t in texts:
        stream += [BOS, *encode(t), EOS]
        if len(stream) > max_tokens:
            break
    total_nll, total = 0.0, 0
    for i in range(0, len(stream) - 1, block):
        chunk = stream[i : i + block + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)
        logits, _ = model(x)
        total_nll += float(F.cross_entropy(logits[0], y[0], reduction="sum"))
        total += y.numel()
    return math.exp(total_nll / max(total, 1))


def completion(model: GPT, ex: InstructExample, max_new: int) -> str:
    return decode(model.generate([BOS, *encode(ex.prompt), SEP], max_new, eos=EOS))


def constraint_rate(model: GPT, examples: list[InstructExample], max_new: int = 200) -> float:
    """Fraction of greedy completions satisfying their Words constraint."""
    if not examples:
        raise ValueError("no examples")
    return sum(satisfies(ex, completion(model, ex, max_new)) for ex in examples) / len(examples)


@dataclass(frozen=True)
class SpeedReport:
    ttft_ms: float  # prefill + first token, median over runs
    decode_tokens_per_s: float  # after the first token, median over runs
    kv_cache_bytes: int  # at the end of generation
    new_tokens: int


@torch.no_grad()
def generation_speed(model: GPT, prompt: list[int], new_tokens: int, runs: int = 5) -> SpeedReport:
    """Time prefill and incremental decoding (fixed length, no EOS stop)."""
    model.eval()
    device = model.tok_emb.weight.device
    ttfts, rates, kv_bytes = [], [], 0
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        logits, cache = model(torch.tensor([prompt], device=device))
        nxt = int(logits[0, -1].argmax())
        t1 = time.perf_counter_ns()
        for _ in range(new_tokens - 1):
            logits, cache = model(torch.tensor([[nxt]], device=device), cache)
            nxt = int(logits[0, -1].argmax())
        t2 = time.perf_counter_ns()
        ttfts.append((t1 - t0) / 1e6)
        rates.append((new_tokens - 1) / max((t2 - t1) / 1e9, 1e-12))
        kv_bytes = sum(
            k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache
        )
    return SpeedReport(float(np.median(ttfts)), float(np.median(rates)), kv_bytes, new_tokens)


@dataclass(frozen=True)
class SpeculativeResult:
    tokens: list[int]
    proposed: int
    accepted: int
    target_forwards: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposed, 1)


@torch.no_grad()
def speculative_generate(
    target: GPT, draft: GPT, prompt: list[int], max_new: int, k: int = 4, eos: int | None = None
) -> SpeculativeResult:
    """Greedy speculative decoding: the draft proposes k tokens, the target verifies them
    in one forward pass and keeps the longest agreeing prefix plus its own next token.
    For clarity this recomputes over the full sequence each round (no cache reuse);
    it measures acceptance, and correctness vs plain greedy, not a tuned speedup."""
    target.eval()
    draft.eval()
    device = target.tok_emb.weight.device
    seq = list(prompt)
    out: list[int] = []
    proposed = accepted = forwards = 0
    while len(out) < max_new and len(seq) < target.cfg.block:
        n = min(k, max_new - len(out), target.cfg.block - len(seq) - 1)
        guesses = draft.generate(seq, n) if n > 0 else []
        logits, _ = target(torch.tensor([seq + guesses], device=device))
        forwards += 1
        preds = logits[0, len(seq) - 1 :].argmax(dim=-1).tolist()  # target's next-token picks
        keep = 0
        while keep < len(guesses) and guesses[keep] == preds[keep]:
            keep += 1
        proposed += len(guesses)
        accepted += keep
        new = guesses[:keep] + [preds[keep]]
        for tok in new:
            if tok == eos or len(out) >= max_new:
                return SpeculativeResult(out, proposed, accepted, forwards)
            out.append(tok)
            seq.append(tok)
    return SpeculativeResult(out, proposed, accepted, forwards)
