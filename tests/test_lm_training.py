from __future__ import annotations

import csv

import pytest
import torch

from edge_ai_compression.lm.data import preference_pairs, synthetic_stories
from edge_ai_compression.lm.eval import (
    constraint_rate,
    generation_speed,
    perplexity,
    speculative_generate,
)
from edge_ai_compression.lm.model import create_gpt
from edge_ai_compression.lm.tokenizer import BOS, encode
from edge_ai_compression.lm.train import LMTrainConfig, dpo_loss, load_lm, seq_logprob, train_lm


def _cfg(tmp_path, stage, **kw):
    base = {
        "stage": stage,
        "model": "gpt_tiny",
        "limit": 300,
        "batch_size": 8,
        "steps": 60,
        "lr": 3e-3,
        "warmup_steps": 5,
        "checkpoint_out": str(tmp_path / f"{stage}.pt"),
        "results_dir": str(tmp_path / "r"),
    }
    return LMTrainConfig.from_dict({**base, **kw})


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lm")
    row = train_lm(_cfg(tmp, "pretrain", steps=150))
    return tmp, row


def test_pretraining_lowers_perplexity_and_logs_run(pretrained):
    tmp, row = pretrained
    texts = [e.story for e in synthetic_stories(20, seed=99)]
    fresh = perplexity(create_gpt("gpt_tiny"), texts)
    trained = perplexity(load_lm(row["checkpoint"])[0], texts)
    assert trained < 0.5 * fresh
    with open(tmp / "r" / "lm_runs.csv", newline="") as f:
        assert next(csv.DictReader(f))["stage"] == "pretrain"


def test_sft_then_dpo_chain_parent_ids_and_dpo_raises_margin(pretrained):
    tmp, pre = pretrained
    sft = train_lm(_cfg(tmp, "sft", init_checkpoint=pre["checkpoint"], steps=80))
    assert sft["parent_run_id"] == pre["run_id"]
    policy, _ = load_lm(sft["checkpoint"])
    pair = preference_pairs(synthetic_stories(5, seed=7))[0]

    def margin(m):
        with torch.no_grad():
            return float(seq_logprob(m, pair[0], pair[1]) - seq_logprob(m, pair[0], pair[2]))

    before = margin(policy)
    dpo = train_lm(
        _cfg(
            tmp,
            "dpo",
            init_checkpoint=sft["checkpoint"],
            steps=30,
            batch_size=4,
            lr=1e-3,
            dpo_beta=0.5,
        )
    )
    assert dpo["parent_run_id"] == sft["run_id"]
    assert margin(load_lm(dpo["checkpoint"])[0]) > before


def test_dpo_loss_is_log2_at_initialization():
    m = create_gpt("gpt_tiny")
    pair = preference_pairs(synthetic_stories(3))[0]
    assert float(dpo_loss(m, m, pair, beta=0.1)) == pytest.approx(0.6931, abs=1e-3)


def test_distilled_draft_and_speculative_decoding_is_lossless(pretrained):
    tmp, pre = pretrained
    draft = train_lm(_cfg(tmp, "distill", teacher_checkpoint=pre["checkpoint"], steps=60))
    target, _ = load_lm(pre["checkpoint"])
    small, _ = load_lm(draft["checkpoint"])
    prompt = [BOS, *encode(" Once upon a time")]
    spec = speculative_generate(target, small, prompt, max_new=40, k=4)
    assert spec.tokens == target.generate(prompt, max_new=40)  # identical to plain greedy
    assert 0.0 <= spec.acceptance_rate <= 1.0 and spec.target_forwards <= 40


def test_constraint_rate_and_speed_report(pretrained):
    _, pre = pretrained
    model, _ = load_lm(pre["checkpoint"])
    rate = constraint_rate(model, synthetic_stories(4, seed=3), max_new=60)
    assert 0.0 <= rate <= 1.0
    rep = generation_speed(model, [BOS, *encode(" Once")], new_tokens=16, runs=2)
    assert rep.ttft_ms > 0 and rep.decode_tokens_per_s > 0 and rep.kv_cache_bytes > 0


def test_config_validation():
    with pytest.raises(ValueError, match="init_checkpoint"):
        LMTrainConfig.from_dict({"stage": "sft"})
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        LMTrainConfig.from_dict({"stage": "distill"})
    with pytest.raises(ValueError, match="unknown lm train keys"):
        LMTrainConfig.from_dict({"epochs": 3})
