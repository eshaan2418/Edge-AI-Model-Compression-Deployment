"""Byte-level tokenizer (DECISIONS D8.1): ids 0-255 are UTF-8 bytes, then specials."""

from __future__ import annotations

PAD, BOS, EOS, SEP = 256, 257, 258, 259
VOCAB_SIZE = 260


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")
