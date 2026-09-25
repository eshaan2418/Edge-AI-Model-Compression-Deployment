from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from edge_ai_compression.analysis.diagnostics import full_diagnostic_report
from edge_ai_compression.benchmarking.evaluator import Evaluator, accuracy_on_loader
from edge_ai_compression.compression.pruning.sparsity import weight_sparsity
from edge_ai_compression.core.compression_orders import format_order
from edge_ai_compression.core.experiment import ExperimentConfig, ExperimentResult
from edge_ai_compression.core.pipeline import CompressionPipeline, enabled_stage_names
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.data.loaders import build_loaders
from edge_ai_compression.experiment_db.record import record_from_run
from edge_ai_compression.experiment_db.writer import (
    append_csv_row,
    append_jsonl_line,
    write_artifacts,
)
from edge_ai_compression.theory.model_complexity import (
    count_parameters,
    estimate_flops_macs,
    layer_gemm_shapes,
)
from edge_ai_compression.utils.config_loader import load_yaml
from edge_ai_compression.utils.reproducibility import set_seed


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    return ExperimentConfig.from_dict(raw)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> ExperimentResult:
        cfg = self.config
        set_seed(cfg.seed)
        train_loader, test_loader = build_loaders(
            cfg.dataset,
            cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            limit_samples=cfg.limit_samples,
        )

        model = ModelRegistry.create(cfg.model, num_classes=cfg.num_classes).to(cfg.device)
        source_run_id: str | None = None
        source_step: int | None = None
        if cfg.checkpoint_in:
            if not os.path.isfile(cfg.checkpoint_in):
                raise FileNotFoundError(
                    f"checkpoint_in {cfg.checkpoint_in!r} does not exist; refusing to "
                    "silently evaluate randomly initialized weights"
                )
            ckpt = torch.load(cfg.checkpoint_in, map_location=cfg.device)
            model.load_state_dict(ckpt["model_state"])
            source_run_id, source_step = ckpt.get("run_id"), ckpt.get("step")

        baseline_sd = _clone_state(model)
        # The baseline only contributes accuracy; its latency belongs to its own run.
        baseline_acc, _ = accuracy_on_loader(model, test_loader, cfg.device)

        pipeline = CompressionPipeline(cfg.compression, order=cfg.compression_order)
        applied_order = format_order(enabled_stage_names(cfg.compression_order, cfg.compression))
        work = ModelRegistry.create(cfg.model, num_classes=cfg.num_classes).to(cfg.device)
        work.load_state_dict({k: v.to(cfg.device) for k, v in baseline_sd.items()})

        work = pipeline.run(
            work,
            train_loader,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            device=cfg.device,
            baseline_accuracy=baseline_acc,
        )

        Path(cfg.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": work.state_dict()}, cfg.checkpoint_out)

        report = Evaluator(cfg.device, cfg.benchmark).evaluate(work, test_loader)

        baseline_model = ModelRegistry.create(cfg.model, num_classes=cfg.num_classes).to(cfg.device)
        baseline_model.load_state_dict({k: v.to(cfg.device) for k, v in baseline_sd.items()})

        db = cfg.experiment_db
        exp_id = str(uuid.uuid4())
        diag = full_diagnostic_report(
            baseline_model,
            work,
            test_loader,
            cfg.device,
            cfg.num_classes,
            max_batches=db.diagnostics_max_batches,
        )
        failure_shift = {
            "per_class_delta": {
                k: diag["per_class_accuracy_compressed"].get(k, 0.0)
                - diag["per_class_accuracy_baseline"].get(k, 0.0)
                for k in diag["per_class_accuracy_baseline"]
            },
            "ece_delta": diag["ece_compressed"] - diag["ece_baseline"],
        }
        metrics: dict[str, Any] = {
            "experiment_id": exp_id,
            "compression_order": applied_order,
            "hardware_profile": cfg.hardware_profile,
            "baseline_accuracy": baseline_acc,
            "benchmark": report.to_dict(),
            "diagnostics": diag,
            "failure_class_shift": failure_shift,
        }

        nparams = count_parameters(work)
        flops = estimate_flops_macs(work, report.input_shape)
        metrics["layer_shapes"] = layer_gemm_shapes(work, (1, *report.input_shape[1:]))

        prune_sec = cfg.compression.pruning
        pruning_type = prune_sec.tag if prune_sec.enabled else "none"
        pruning_sparsity = prune_sec.target_sparsity if prune_sec.enabled else 0.0

        record = record_from_run(
            experiment_id=exp_id,
            model_name=cfg.model,
            dataset=cfg.dataset,
            num_params=nparams,
            flops=flops,
            compression_order=applied_order,
            pruning_type=pruning_type,
            pruning_sparsity=pruning_sparsity,
            weight_sparsity=weight_sparsity(work),
            quantization_type=cfg.compression.quantization.tag
            if cfg.compression.quantization.enabled
            else "none",
            distillation_enabled=cfg.compression.distillation.enabled,
            temperature=float(cfg.compression.distillation.temperature),
            alpha=float(cfg.compression.distillation.alpha),
            device=f"{cfg.device}:{cfg.hardware_profile}",
            baseline_accuracy=baseline_acc,
            report=report,
            source_run_id=source_run_id,
            source_step=source_step,
            extra={"failure_shift": failure_shift},
        )
        append_csv_row(record, db.path)
        append_jsonl_line(record.to_json(), db.path)
        write_artifacts(
            exp_id,
            db.path,
            config_dict=cfg.to_dict(),
            metrics=metrics,
            model=work,
            report=report,
            confusion_matrix=np.array(diag["confusion_matrix_compressed"], dtype=np.int64),
            failure_cases=diag.get("failure_cases"),
        )

        return ExperimentResult(
            accuracy=report.accuracy,
            latency_ms_mean=report.latency.mean,
            latency_ms_p99=report.latency.p99,
            size_mb=report.size_mb,
            peak_ram_mib=report.peak_rss_mib,
            extras={
                "experiment_id": exp_id,
                "baseline_accuracy": baseline_acc,
                "latency_median_ms": report.latency_median_ms,
                "cold_start_ms": report.cold_start_ms,
                "environment_problems": report.environment_problems,
            },
        )
