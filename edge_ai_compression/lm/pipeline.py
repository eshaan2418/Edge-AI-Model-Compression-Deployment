"""Small-LM lifecycle end to end from one YAML config (resumable).

pretrain -> SFT -> DPO, a distilled draft for speculative decoding, then the
quantization ladder on each stage's checkpoint, logged to ``lm_evals.csv``. Stages
whose checkpoint exists are skipped; ladder rows already logged are skipped.

    python -m edge_ai_compression.lm.pipeline --config edge_ai_compression/configs/lm/smoke.yml
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from edge_ai_compression.lm.compress import (
    LM_LADDER,
    evaluate,
    log_eval,
    quantize_gpt,
    speculative_row,
)
from edge_ai_compression.lm.data import load_instruct, load_tinystories, synthetic_stories
from edge_ai_compression.lm.tokenizer import BOS, encode
from edge_ai_compression.lm.train import LMTrainConfig, load_lm, train_lm
from edge_ai_compression.utils.config_loader import load_yaml


@dataclass(frozen=True)
class PipelineConfig:
    size: str = "gpt_10m"
    draft: str = "gpt_tiny"
    data: str = "synthetic"
    stories_path: str | None = None
    instruct_path: str | None = None
    eval_stories_path: str | None = None  # held-out stories for perplexity (default: train split)
    eval_instruct_path: str | None = None
    limit: int | None = None
    device: str = "cpu"
    seed: int = 0
    batch_size: int = 32
    lr: float = 3e-4
    steps: dict[str, int] = field(
        default_factory=lambda: {"pretrain": 1000, "sft": 200, "dpo": 100, "distill": 500}
    )
    eval_stories: int = 200
    constraint_examples: int = 100
    max_new: int = 200
    methods: tuple[str, ...] = tuple(LM_LADDER)
    speculative_k: int = 4
    results_dir: str = "results"
    models_dir: str = "models/lm"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PipelineConfig:
        unknown = set(d) - {f.name for f in fields(PipelineConfig)}
        if unknown:
            raise ValueError(f"unknown lm pipeline keys: {sorted(unknown)}")
        d = dict(d)
        if "methods" in d:
            d["methods"] = tuple(d["methods"])
            bad = set(d["methods"]) - set(LM_LADDER)
            if bad:
                raise ValueError(f"unknown ladder methods {sorted(bad)}")
        return PipelineConfig(**d)


def _stage_cfg(cfg: PipelineConfig, stage: str, out: Path, **kw: Any) -> LMTrainConfig:
    return LMTrainConfig(
        stage=stage,
        model=kw.pop("model", cfg.size),
        data=cfg.data,
        stories_path=cfg.stories_path,
        instruct_path=cfg.instruct_path,
        limit=cfg.limit,
        batch_size=cfg.batch_size,
        steps=cfg.steps[stage],
        lr=kw.pop("lr", cfg.lr),
        warmup_steps=max(1, cfg.steps[stage] // 20),
        device=cfg.device,
        seed=cfg.seed,
        checkpoint_out=str(out),
        results_dir=cfg.results_dir,
        **kw,
    )


def _logged(results: Path) -> set[tuple[str, str]]:
    path = results / "lm_evals.csv"
    if not path.is_file():
        return set()
    with open(path, newline="") as f:
        return {(r["source_run_id"], r["method"]) for r in csv.DictReader(f)}


def run_pipeline(cfg: PipelineConfig) -> None:
    results, models = Path(cfg.results_dir), Path(cfg.models_dir) / f"{cfg.size}_s{cfg.seed}"
    ckpt = {s: models / f"{s}.pt" for s in ("pretrain", "sft", "dpo", "draft")}
    if not ckpt["pretrain"].is_file():
        train_lm(_stage_cfg(cfg, "pretrain", ckpt["pretrain"]))
    if not ckpt["sft"].is_file():
        train_lm(_stage_cfg(cfg, "sft", ckpt["sft"], init_checkpoint=str(ckpt["pretrain"])))
    if not ckpt["dpo"].is_file():
        train_lm(
            _stage_cfg(cfg, "dpo", ckpt["dpo"], init_checkpoint=str(ckpt["sft"]), lr=cfg.lr / 10)
        )
    if not ckpt["draft"].is_file():
        train_lm(
            _stage_cfg(
                cfg,
                "distill",
                ckpt["draft"],
                model=cfg.draft,
                teacher_checkpoint=str(ckpt["pretrain"]),
            )
        )

    if cfg.eval_stories_path:
        eval_texts = load_tinystories(Path(cfg.eval_stories_path), cfg.eval_stories)
    elif cfg.data == "synthetic":
        eval_texts = [e.story for e in synthetic_stories(cfg.eval_stories, cfg.seed + 1000)]
    else:
        eval_texts = load_tinystories(Path(str(cfg.stories_path)), cfg.eval_stories)
    if cfg.eval_instruct_path:
        examples = load_instruct(Path(cfg.eval_instruct_path), cfg.constraint_examples)
    elif cfg.data == "synthetic":
        examples = synthetic_stories(cfg.constraint_examples, cfg.seed + 2000)
    else:
        examples = load_instruct(Path(str(cfg.instruct_path)), cfg.constraint_examples)

    done = _logged(results)
    for stage in ("pretrain", "sft", "dpo"):
        base, meta_ckpt = load_lm(ckpt[stage], cfg.device)
        run_id = str(meta_ckpt["run_id"])
        stage_examples = examples if stage != "pretrain" else []
        for method in cfg.methods:
            if (run_id, method) in done:
                continue
            model = quantize_gpt(copy.deepcopy(base), method, eval_texts, cfg.device)
            meta = {
                "source_run_id": run_id,
                "stage": stage,
                "model": cfg.size,
                "seed": cfg.seed,
                "method": method,
                "device": cfg.device,
            }
            log_eval(results, meta, evaluate(model, eval_texts, stage_examples, cfg.max_new))
    target, meta_ckpt = load_lm(ckpt["pretrain"], cfg.device)
    if (str(meta_ckpt["run_id"]), "speculative") not in done:
        draft, _ = load_lm(ckpt["draft"], cfg.device)
        prompt = [BOS, *encode(" Once upon a time")]
        meta = {
            "source_run_id": str(meta_ckpt["run_id"]),
            "stage": "pretrain",
            "model": cfg.size,
            "seed": cfg.seed,
            "method": "speculative",
            "device": cfg.device,
        }
        log_eval(results, meta, speculative_row(target, draft, prompt, 64, cfg.speculative_k))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    from edge_ai_compression.utils.overrides import apply_overrides, parse_overrides

    run_pipeline(
        PipelineConfig.from_dict(apply_overrides(load_yaml(a.config), parse_overrides(a.set)))
    )


if __name__ == "__main__":
    main()
