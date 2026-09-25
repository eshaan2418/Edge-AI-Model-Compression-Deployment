"""Exceptions and table reading shared by the Phase 6 analyses."""

from __future__ import annotations

import pandas as pd


class InsufficientData(ValueError):
    """Too few runs / configurations for an analysis (distinct from code errors)."""


def read_table(path: object) -> pd.DataFrame:
    """Read a DB CSV with id columns as strings (an all-empty id column would
    otherwise be parsed as float and break joins)."""
    ids = ("experiment_id", "run_id", "source_run_id", "fingerprint_hash", "git_commit")
    return pd.read_csv(path, dtype={c: "string" for c in ids})
