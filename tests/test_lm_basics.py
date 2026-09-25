from __future__ import annotations

import pytest
import torch

from edge_ai_compression.lm.data import (
    SFTData,
    TokenWindows,
    parse_instruct,
    preference_pairs,
    satisfies,
    synthetic_stories,
)
from edge_ai_compression.lm.model import GPT_SIZES, create_gpt, num_params
from edge_ai_compression.lm.tokenizer import BOS, PAD, SEP, decode, encode


def test_tokenizer_roundtrip_including_unicode():
    text = "Once upon a time, café 🐶"
    assert decode(encode(text)) == text
    assert decode([BOS, *encode("hi"), SEP]) == "hi"


def test_synthetic_examples_satisfy_their_constraint_and_pairs_violate_it():
    exs = synthetic_stories(20)
    assert all(satisfies(e, e.story) for e in exs)
    pairs = preference_pairs(exs)
    assert len(pairs) >= 18
    for (prompt, chosen, rejected), ex in zip(pairs, exs, strict=False):
        assert prompt == ex.prompt and satisfies(ex, chosen) and not satisfies(ex, rejected)


def test_parse_instruct_record():
    rec = (
        "Features: Dialogue\nWords: jump, red, happy\nSummary: A dog jumps.\n"
        "Story: The red dog was happy to jump.\n"
    )
    ex = parse_instruct(rec)
    assert ex.words == ("jump", "red", "happy") and satisfies(ex, ex.story)
    assert parse_instruct("no fields here") is None


def test_sft_targets_mask_the_prompt():
    ex = synthetic_stories(1)[0]
    x, y = SFTData([ex], block=256)[0]
    prompt_len = len(encode(ex.prompt)) + 2  # BOS ... SEP
    assert (y[: prompt_len - 1] == PAD).all() and y[prompt_len - 1] != PAD
    assert x.shape == y.shape == (256,)


def test_token_windows_shift_by_one():
    ds = TokenWindows([e.story for e in synthetic_stories(50)], block=32)
    x, y = ds[0]
    assert torch.equal(x[1:], y[:-1])


@pytest.mark.parametrize("name,target", [("gpt_10m", 10e6), ("gpt_25m", 25e6), ("gpt_50m", 50e6)])
def test_model_sizes(name, target):
    assert abs(num_params(create_gpt(name)) - target) / target < 0.15


def test_kv_cache_generation_matches_full_recompute():
    torch.manual_seed(0)
    m = create_gpt("gpt_tiny").eval()
    prompt = [BOS, *encode("Once upon")]
    out = m.generate(prompt, max_new=20)
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(20):
            logits, _ = m(torch.tensor([seq]))
            seq.append(int(logits[0, -1].argmax()))
    assert out == seq[len(prompt) :]
    with pytest.raises(ValueError, match="context length"):
        m(torch.zeros(1, GPT_SIZES["gpt_tiny"].block + 1, dtype=torch.long))
