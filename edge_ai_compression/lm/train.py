"""Training stages of the small-LM lifecycle, logged to ``lm_runs.csv``.

stage:
- ``pretrain``: next-token cross-entropy on story windows
- ``sft``: instruction -> story, loss on response tokens only
- ``distill``: student learns from a teacher's logits (KD at temperature T, mixed
  with CE); used to build the speculative-decoding draft model
- ``dpo``: Direct Preference Optimization (Rafailov et al. 2023) on
  constraint-satisfying vs constraint-violating stories, reference = frozen copy of
  the initial (SFT) policy

    python -m edge_ai_compression.lm.train --config edge_ai_compression/configs/lm/<stage>.yml
"""

from __future__ import annotations

import argparse
import copy
import math
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edge_ai_compression.benchmarking.fingerprint import collect
from edge_ai_compression.lm.data import (
    InstructExample,
    SFTData,
    TokenWindows,
    load_instruct,
    load_tinystories,
    preference_pairs,
    synthetic_stories,
)
from edge_ai_compression.lm.eval import perplexity
from edge_ai_compression.lm.model import GPT, GPT_SIZES, create_gpt, num_params
from edge_ai_compression.lm.tokenizer import BOS, EOS, PAD, SEP, encode
from edge_ai_compression.utils.config_loader import load_yaml
from edge_ai_compression.utils.reproducibility import set_seed

STAGES = ("pretrain", "sft", "distill", "dpo")


@dataclass(frozen=True)
class LMTrainConfig:
    stage: str = "pretrain"
    model: str = "gpt_10m"  # size of a fresh model (ignored when init_checkpoint is set)
    init_checkpoint: str | None = None  # start from this checkpoint (sft / dpo)
    teacher_checkpoint: str | None = None  # distill
    data: str = "synthetic"  # synthetic | tinystories
    stories_path: str | None = None  # TinyStories txt (pretrain / distill / eval)
    instruct_path: str | None = None  # TinyStories-Instruct txt (sft / dpo)
    limit: int | None = None  # cap number of stories / examples
    val_fraction: float = 0.02
    batch_size: int = 32
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    kd_temperature: float = 2.0
    kd_alpha: float = 0.9
    dpo_beta: float = 0.1
    device: str = "cpu"
    seed: int = 0
    checkpoint_out: str = "models/lm/model.pt"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage '{self.stage}'; expected {STAGES}")
        if self.data not in ("synthetic", "tinystories"):
            raise ValueError("data must be synthetic or tinystories")
        if self.model not in GPT_SIZES:
            raise ValueError(f"unknown model size '{self.model}'")
        if self.stage in ("sft", "dpo") and not self.init_checkpoint:
            raise ValueError(f"{self.stage} needs init_checkpoint")
        if self.stage == "distill" and not self.teacher_checkpoint:
            raise ValueError("distill needs teacher_checkpoint")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LMTrainConfig:
        unknown = set(d) - {f.name for f in fields(LMTrainConfig)}
        if unknown:
            raise ValueError(f"unknown lm train keys: {sorted(unknown)}")
        return LMTrainConfig(**d)


def load_lm(path: str | Path, device: str = "cpu") -> tuple[GPT, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_gpt(ckpt["size"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def stories(cfg: LMTrainConfig) -> list[str]:
    if cfg.data == "synthetic":
        return [e.story for e in synthetic_stories(cfg.limit or 2000, cfg.seed)]
    if not cfg.stories_path:
        raise ValueError("tinystories data needs stories_path")
    return load_tinystories(Path(cfg.stories_path), cfg.limit)


def instruct(cfg: LMTrainConfig) -> list[InstructExample]:
    if cfg.data == "synthetic":
        return synthetic_stories(cfg.limit or 2000, cfg.seed)
    if not cfg.instruct_path:
        raise ValueError("tinystories data needs instruct_path")
    return load_instruct(Path(cfg.instruct_path), cfg.limit)


def split(items: list[Any], val_fraction: float) -> tuple[list[Any], list[Any]]:
    n_val = max(1, int(len(items) * val_fraction))
    return items[n_val:], items[:n_val]


def seq_logprob(model: GPT, prompt: str, response: str) -> torch.Tensor:
    """Sum of log p(response tokens | prompt) under ``model`` (differentiable)."""
    p_ids = [BOS, *encode(prompt), SEP]
    r_ids = [*encode(response), EOS]
    ids = (p_ids + r_ids)[: model.cfg.block + 1]
    x = torch.tensor([ids[:-1]], device=model.tok_emb.weight.device)
    y = torch.tensor([ids[1:]], device=x.device)
    logp = F.log_softmax(model(x)[0][0], dim=-1).gather(1, y[0].unsqueeze(1)).squeeze(1)
    return logp[len(p_ids) - 1 :].sum()


def dpo_loss(policy: GPT, ref: GPT, pair: tuple[str, str, str], beta: float) -> torch.Tensor:
    prompt, chosen, rejected = pair
    with torch.no_grad():
        ref_c, ref_r = seq_logprob(ref, prompt, chosen), seq_logprob(ref, prompt, rejected)
    pol_c, pol_r = seq_logprob(policy, prompt, chosen), seq_logprob(policy, prompt, rejected)
    return -F.logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r)))


def _lr(step: int, cfg: LMTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * progress))


def _batches(loader: DataLoader) -> Iterator[list[torch.Tensor]]:
    while True:
        yield from loader


def train_lm(cfg: LMTrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    fingerprint = collect()
    device = cfg.device
    parent = None
    if cfg.init_checkpoint:
        model, ckpt = load_lm(cfg.init_checkpoint, device)
        size, parent = ckpt["size"], ckpt.get("run_id")
    else:
        size = cfg.model
        model = create_gpt(size).to(device)
    teacher = load_lm(cfg.teacher_checkpoint, device)[0].eval() if cfg.teacher_checkpoint else None
    ref = copy.deepcopy(model).eval() if cfg.stage == "dpo" else None

    texts_train, texts_val = split(stories(cfg), cfg.val_fraction)
    if cfg.stage in ("sft", "dpo"):
        ex_train, ex_val = split(instruct(cfg), cfg.val_fraction)
        texts_val = [e.story for e in ex_val]
    loader = None
    pairs: list[tuple[str, str, str]] = []
    if cfg.stage in ("pretrain", "distill"):
        loader = DataLoader(
            TokenWindows(texts_train, model.cfg.block, cfg.seed),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
    elif cfg.stage == "sft":
        loader = DataLoader(
            SFTData(ex_train, model.cfg.block), batch_size=cfg.batch_size, shuffle=True
        )
    else:
        pairs = preference_pairs(ex_train, cfg.seed)
        if not pairs:
            raise ValueError("no preference pairs")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    batches = _batches(loader) if loader is not None else None
    t0 = time.perf_counter()
    model.train()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr(step, cfg)
        if cfg.stage == "dpo":
            chunk = [pairs[(step * cfg.batch_size + i) % len(pairs)] for i in range(cfg.batch_size)]
            loss = torch.stack([dpo_loss(model, ref, p, cfg.dpo_beta) for p in chunk]).mean()
        else:
            assert batches is not None
            x, y = (t.to(device) for t in next(batches))
            logits, _ = model(x)
            ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=PAD)
            loss = ce
            if cfg.stage == "distill":
                assert teacher is not None
                with torch.no_grad():
                    t_logits, _ = teacher(x)
                t = cfg.kd_temperature
                kd = F.kl_div(
                    F.log_softmax(logits / t, -1).flatten(0, 1),
                    F.softmax(t_logits / t, -1).flatten(0, 1),
                    reduction="batchmean",
                )
                loss = cfg.kd_alpha * kd * t * t + (1 - cfg.kd_alpha) * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_seconds = time.perf_counter() - t0
    val_ppl = perplexity(model, texts_val)

    run_id = str(uuid.uuid4())
    Path(cfg.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "size": size,
            "run_id": run_id,
            "stage": cfg.stage,
            "config": asdict(cfg),
        },
        cfg.checkpoint_out,
    )
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "git_commit": fingerprint.git_commit,
        "git_dirty": fingerprint.git_dirty,
        "fingerprint_hash": fingerprint.hash,
        "stage": cfg.stage,
        "model": size,
        "params": num_params(model),
        "parent_run_id": parent,
        "steps": cfg.steps,
        "seed": cfg.seed,
        "data": cfg.data,
        "val_ppl": val_ppl,
        "train_seconds": train_seconds,
        "checkpoint": cfg.checkpoint_out,
    }
    from edge_ai_compression.experiment_db.writer import append_row

    append_row(Path(cfg.results_dir) / "lm_runs.csv", tuple(row), row)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    from edge_ai_compression.utils.overrides import apply_overrides, parse_overrides

    cfg = apply_overrides(load_yaml(a.config), parse_overrides(a.set))
    row = train_lm(LMTrainConfig.from_dict(cfg))
    print({k: row[k] for k in ("run_id", "stage", "model", "val_ppl", "checkpoint")})


if __name__ == "__main__":
    main()
