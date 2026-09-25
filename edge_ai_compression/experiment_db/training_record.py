"""Rows of ``training_runs.csv``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge_ai_compression.benchmarking.fingerprint import Fingerprint
from edge_ai_compression.pretraining.config import TrainConfig

TRAINING_SCHEMA_VERSION = 1


@dataclass
class TrainingRecord:
    run_id: str
    timestamp: str
    schema_version: int
    git_commit: str | None
    git_dirty: bool | None
    fingerprint_hash: str
    model: str
    dataset: str
    variant: str
    seed: int
    epochs: int
    steps: int
    lr: float
    batch_size: int
    device: str
    test_accuracy: float
    test_loss: float
    train_seconds: float
    final_checkpoint: str

    @staticmethod
    def create(
        *,
        run_id: str,
        cfg: TrainConfig,
        fingerprint: Fingerprint,
        steps: int,
        test_accuracy: float,
        test_loss: float,
        train_seconds: float,
        final_checkpoint: str,
    ) -> TrainingRecord:
        return TrainingRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            schema_version=TRAINING_SCHEMA_VERSION,
            git_commit=fingerprint.git_commit,
            git_dirty=fingerprint.git_dirty,
            fingerprint_hash=fingerprint.hash,
            model=cfg.model,
            dataset=cfg.dataset,
            variant=cfg.variant,
            seed=cfg.seed,
            epochs=cfg.epochs,
            steps=steps,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            device=cfg.device,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            train_seconds=train_seconds,
            final_checkpoint=final_checkpoint,
        )

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)


TRAINING_CSV_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TrainingRecord))


def append_training_row(record: TrainingRecord, results_dir: Path) -> None:
    from edge_ai_compression.experiment_db.writer import append_row

    append_row(results_dir / "training_runs.csv", TRAINING_CSV_FIELDS, record.to_csv_row())


SIGNAL_CSV_FIELDS: tuple[str, ...] = ("run_id", "step", "epoch")


def signal_fields() -> tuple[str, ...]:
    from edge_ai_compression.pretraining.signals import SCALAR_FIELDS

    return SIGNAL_CSV_FIELDS + SCALAR_FIELDS


def append_signal_rows(rows: list[dict[str, Any]], results_dir: Path) -> None:
    """Append rows of ``training_signals.csv`` (one per logged step)."""
    from edge_ai_compression.experiment_db.writer import append_row

    fields = signal_fields()
    for row in rows:
        append_row(results_dir / "training_signals.csv", fields, {k: row[k] for k in fields})
