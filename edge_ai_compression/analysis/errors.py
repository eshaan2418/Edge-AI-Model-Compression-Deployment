"""Exceptions shared by the Phase 6 analyses."""

from __future__ import annotations


class InsufficientData(ValueError):
    """Too few runs / configurations for an analysis (distinct from code errors)."""
