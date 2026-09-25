"""Decoder-only GPT with a KV cache.

Pre-norm blocks with explicit ``qkv`` / ``proj`` / ``fc1`` / ``fc2`` Linear layers so
the Phase 3 quantizers apply unchanged; tied token embedding / LM head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_ai_compression.lm.tokenizer import VOCAB_SIZE

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class GPTConfig:
    dim: int = 256
    depth: int = 12
    heads: int = 4
    block: int = 512
    vocab: int = VOCAB_SIZE
    mlp_mult: int = 4


GPT_SIZES = {
    "gpt_tiny": GPTConfig(dim=64, depth=2, heads=2, block=128),  # tests / smoke only
    "gpt_10m": GPTConfig(dim=256, depth=12, heads=4),
    "gpt_25m": GPTConfig(dim=384, depth=14, heads=6),
    "gpt_50m": GPTConfig(dim=512, depth=16, heads=8),
}


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(
        self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        p = k.shape[2] - t  # cached positions before this chunk
        if p == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.ones(t, p + t, dtype=torch.bool, device=x.device).tril(diagonal=p)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, t, -1)), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.fc1 = nn.Linear(cfg.dim, cfg.mlp_mult * cfg.dim)
        self.fc2 = nn.Linear(cfg.mlp_mult * cfg.dim, cfg.dim)

    def forward(
        self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        a, kv = self.attn(self.ln1(x), past)
        x = x + a
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)))), kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(
        self, idx: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        past_len = cache[0][0].shape[2] if cache else 0
        if past_len + idx.shape[1] > self.cfg.block:
            raise ValueError("sequence exceeds the model's context length")
        pos = torch.arange(past_len, past_len + idx.shape[1], device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, cache[i] if cache else None)
            new_cache.append(kv)
        return self.lm_head(self.ln_f(x)), new_cache

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int, eos: int | None = None) -> list[int]:
        """Greedy decoding with the KV cache; returns only the new tokens."""
        self.eval()
        device = self.tok_emb.weight.device
        logits, cache = self(torch.tensor([prompt], device=device))
        out: list[int] = []
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax())
            if nxt == eos:
                break
            out.append(nxt)
            if len(prompt) + len(out) >= self.cfg.block:
                break
            logits, cache = self(torch.tensor([[nxt]], device=device), cache)
        return out


def create_gpt(name: str) -> GPT:
    if name not in GPT_SIZES:
        raise KeyError(f"unknown GPT size '{name}'; expected {sorted(GPT_SIZES)}")
    return GPT(GPT_SIZES[name])


def num_params(model: GPT) -> int:
    """Unique parameters (tied embedding counted once), excluding position embeddings."""
    seen = {id(p): p for p in model.parameters()}
    return sum(p.numel() for p in seen.values()) - model.pos_emb.weight.numel()
