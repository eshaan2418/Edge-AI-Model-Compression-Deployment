from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from edge_ai_compression.analysis.diagnostics import full_diagnostic_report
from edge_ai_compression.benchmarking.evaluator import Evaluator, ResultLogger
from edge_ai_compression.core.compression_orders import format_order
from edge_ai_compression.core.experiment import ExperimentConfig, ExperimentResult
from edge_ai_compression.core.pipeline import CompressionPipeline
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.data.loaders import build_loaders
from edge_ai_compression.experiment_db.record import record_from_run
from edge_ai_compression.experiment_db.writer import append_csv_row, append_jsonl_line, write_artifacts
from edge_ai_compression.theory.model_complexity import count_parameters, estimate_flops_macs
from edge_ai_compression.utils.config_loader import load_yaml
from edge_ai_compression.utils.logger import append_jsonl
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
            cfg.dataset, cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )

        model = ModelRegistry.create(cfg.model, num_classes=cfg.num_classes).to(cfg.device)
        if cfg.checkpoint_in and os.path.isfile(cfg.checkpoint_in):
            ckpt = torch.load(cfg.checkpoint_in, map_location=cfg.device)
            model.load_state_dict(ckpt["model_state"])

        baseline_sd = _clone_state(model)
        bench_cfg = cfg.benchmark
        evaluator = Evaluator(
            device=cfg.device,
            latency_repeats=int(bench_cfg.get("latency_repeats", 80)),
            latency_warmup=int(bench_cfg.get("latency_warmup", 3)),
        )
        baseline_report = evaluator.evaluate(model, test_loader)
        baseline_acc = baseline_report.accuracy

        pipeline = CompressionPipeline(cfg.compression, order=cfg.compression_order)
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

        report = evaluator.evaluate(work, test_loader)
        extras: dict[str, Any] = dict(report.extras)

        baseline_model = ModelRegistry.create(cfg.model, num_classes=cfg.num_classes).to(cfg.device)
        baseline_model.load_state_dict({k: v.to(cfg.device) for k, v in baseline_sd.items()})

        db_cfg = cfg.experiment_db
        exp_id = str(uuid.uuid4())
        diag_max = int(db_cfg.get("diagnostics_max_batches", 15))
        metrics: dict[str, Any] = {
            "experiment_id": exp_id,
            "compression_order": format_order(cfg.compression_order),
            "hardware_profile": cfg.hardware_profile,
            "baseline_accuracy": baseline_acc,
            "compressed": report.to_dict(),
            "latency_p50": report.extras.get("latency", {}).get("p50"),
            "latency_p90": report.extras.get("latency", {}).get("p90"),
        }

        if db_cfg.get("enabled", False):
            diag = full_diagnostic_report(
                baseline_model,
                work,
                test_loader,
                cfg.device,
                cfg.num_classes,
                max_batches=diag_max,
            )
            metrics["diagnostics"] = diag
            failure_shift = {
                "per_class_delta": {
                    k: diag["per_class_accuracy_compressed"].get(k, 0.0)
                    - diag["per_class_accuracy_baseline"].get(k, 0.0)
                    for k in diag["per_class_accuracy_baseline"]
                },
                "ece_delta": diag["ece_compressed"] - diag["ece_baseline"],
            }
            metrics["failure_class_shift"] = failure_shift
            extras["failure_class_shift"] = failure_shift

        nparams = count_parameters(work)
        shape = cfg.benchmark.get("input_shape", [1, 3, 32, 32])
        shape_t = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
        flops = estimate_flops_macs(work, shape_t)

        pruning_type = (
            f"{cfg.compression.pruning.mode}:{cfg.compression.pruning.scorer}"
            if cfg.compression.pruning.enabled
            else "none"
        )
        pruning_sparsity = float(cfg.compression.pruning.amount) if cfg.compression.pruning.enabled else 0.0

        result = ExperimentResult(
            accuracy=report.accuracy,
            latency_ms_mean=report.latency_ms_mean,
            latency_ms_p99=report.latency_ms_p99,
            size_mb=report.size_mb,
            peak_ram_mib=report.peak_ram_mib,
            extras={
                "energy_proxy": report.energy_proxy,
                "baseline_accuracy": baseline_acc,
                **extras,
            },
        )

        log_path = bench_cfg.get("results_md", "edge_ai_compression/results/benchmark_append.md")
        ResultLogger.log_md(
            log_path,
            name=cfg.model,
            row={
                "size_mb": report.size_mb,
                "latency_ms_mean": report.latency_ms_mean,
                "latency_ms_p99": report.latency_ms_p99,
                "peak_ram_mib": report.peak_ram_mib,
                "accuracy": report.accuracy,
                "energy_proxy": report.energy_proxy,
            },
        )
        jsonl_legacy = bench_cfg.get("results_jsonl", "edge_ai_compression/results/runs.jsonl")
        append_jsonl(jsonl_legacy, {"config": asdict(cfg), "result": result.to_dict()})

        if db_cfg.get("enabled", False):
            trace = np.asarray(extras.get("latency_trace_ms", []), dtype=np.float64)
            cm = np.array(metrics["diagnostics"]["confusion_matrix_compressed"], dtype=np.int64)
            record = record_from_run(
                experiment_id=exp_id,
                model_name=cfg.model,
                dataset=cfg.dataset,
                num_params=nparams,
                flops=flops,
                compression_order=format_order(cfg.compression_order),
                pruning_type=pruning_type,
                pruning_sparsity=pruning_sparsity,
                quantization_type=cfg.compression.quantization.mode
                if cfg.compression.quantization.enabled
                else "none",
                distillation_enabled=cfg.compression.distillation.enabled,
                temperature=float(cfg.compression.distillation.temperature),
                alpha=float(cfg.compression.distillation.alpha),
                device=f"{cfg.device}:{cfg.hardware_profile}",
                baseline_accuracy=baseline_acc,
                accuracy=report.accuracy,
                latency=report.extras["latency"],
                size_mb=report.size_mb,
                ram_mb=report.peak_ram_mib,
                energy_proxy=report.energy_proxy,
                extra={"failure_shift": metrics.get("failure_class_shift")},
            )
            append_csv_row(record)
            append_jsonl_line(record.to_json())
            write_artifacts(
                exp_id,
                config_dict=asdict(cfg),
                metrics=metrics,
                model=work,
                latency_trace_ms=trace,
                confusion_matrix=cm,
                failure_cases=metrics["diagnostics"].get("failure_cases"),
            )

        return result
