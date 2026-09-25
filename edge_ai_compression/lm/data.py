"""Text data for the small-LM track: TinyStories (+ Instruct) and an offline synthetic set.

Instruction examples use TinyStories-Instruct's ``Words:`` constraint: the story
must contain every listed word. That constraint is the post-trained *behavior*
measured in Phase 8, next to perplexity as raw *capability*.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from edge_ai_compression.lm.tokenizer import BOS, EOS, PAD, SEP, encode

NAMES = ("Lily", "Tom", "Mia", "Ben", "Sam", "Anna", "Max", "Zoe")
ADJS = ("happy", "little", "brave", "shy", "kind", "curious", "sleepy", "funny")
NOUNS = ("dog", "cat", "bird", "ball", "tree", "boat", "cake", "star", "frog", "hat")
VERBS = ("found", "saw", "liked", "wanted", "made", "lost", "shared", "painted")


@dataclass(frozen=True)
class InstructExample:
    words: tuple[str, ...]
    story: str

    @property
    def prompt(self) -> str:
        return f"Words: {', '.join(self.words)}\nStory:"


def synthetic_stories(n: int, seed: int = 0) -> list[InstructExample]:
    """Template stories that contain their three required words (offline tests/smoke)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        name, adj, noun, verb = (rng.choice(x) for x in (NAMES, ADJS, NOUNS, VERBS))
        noun2 = rng.choice([x for x in NOUNS if x != noun])
        story = (
            f" Once upon a time there was a {adj} {noun} named {name}. One day {name} "
            f"{verb} a {noun2}. {name} was very {adj} and played with the {noun2} all day."
        )
        out.append(InstructExample((adj, noun, noun2), story))
    return out


def load_tinystories(path: Path, limit: int | None = None) -> list[str]:
    """TinyStories text file: stories separated by '<|endoftext|>'."""
    stories = [s.strip() for s in path.read_text(encoding="utf-8").split("<|endoftext|>")]
    stories = [s for s in stories if s]
    return stories[:limit] if limit else stories


def parse_instruct(text: str) -> InstructExample | None:
    """One TinyStories-Instruct record ("...Words: a, b, c ... Story: ...")."""
    words = re.search(r"^Words:\s*(.+)$", text, re.MULTILINE)
    story = re.search(r"^Story:\s*(.+)", text, re.MULTILINE | re.DOTALL)
    if not words or not story:
        return None
    ws = tuple(w.strip() for w in words.group(1).split(",") if w.strip())
    return InstructExample(ws, " " + story.group(1).strip()) if ws else None


def load_instruct(path: Path, limit: int | None = None) -> list[InstructExample]:
    records = [parse_instruct(r) for r in path.read_text(encoding="utf-8").split("<|endoftext|>")]
    records = [r for r in records if r is not None]
    return records[:limit] if limit else records


def satisfies(example: InstructExample, text: str) -> bool:
    """All required words appear as whole words (case-insensitive)."""
    return all(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in example.words)


class TokenWindows(Dataset):
    """Random-offset windows of a token stream for next-token pre-training."""

    def __init__(self, texts: list[str], block: int, seed: int = 0) -> None:
        stream: list[int] = []
        for t in texts:
            stream += [BOS, *encode(t), EOS]
        if len(stream) <= block:
            raise ValueError("not enough text for one block")
        self.tokens = torch.tensor(stream, dtype=torch.long)
        self.block = block
        self.n = max(1, (len(stream) - 1) // block)
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(torch.randint(0, len(self.tokens) - self.block - 1, (1,), generator=self.gen))
        chunk = self.tokens[start : start + self.block + 1]
        return chunk[:-1], chunk[1:]


def sft_tensors(ex: InstructExample, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(input ids, target ids) with targets = PAD (ignored) outside the response."""
    prompt = [BOS, *encode(ex.prompt), SEP]
    response = [*encode(ex.story), EOS]
    ids = (prompt + response)[: block + 1]
    targets = ([PAD] * len(prompt) + response)[: block + 1]
    ids += [PAD] * (block + 1 - len(ids))
    targets += [PAD] * (block + 1 - len(targets))
    return torch.tensor(ids[:-1]), torch.tensor(targets[1:])


class SFTData(Dataset):
    def __init__(self, examples: list[InstructExample], block: int) -> None:
        self.items = [sft_tensors(e, block) for e in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[i]


def violate(ex: InstructExample, seed: int = 0) -> str:
    """The story with each required word replaced by a different word (constraint broken)."""
    rng = random.Random(seed)
    story = ex.story
    pool = [w for w in (*ADJS, *NOUNS, "thing", "place", "day") if w not in ex.words]
    for w in ex.words:
        story = re.sub(rf"\b{re.escape(w)}\b", rng.choice(pool), story, flags=re.IGNORECASE)
    return story


def preference_pairs(examples: list[InstructExample], seed: int = 0) -> list[tuple[str, str, str]]:
    """(prompt, chosen, rejected): chosen satisfies the Words constraint, rejected does not."""
    pairs = []
    for i, ex in enumerate(examples):
        rejected = violate(ex, seed + i)
        if rejected != ex.story and not satisfies(ex, rejected):
            pairs.append((ex.prompt, ex.story, rejected))
    return pairs
