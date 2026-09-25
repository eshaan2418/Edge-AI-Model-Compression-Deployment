"""Summary statistics and hypothesis tests for benchmark measurements.

Iterations inside one process are autocorrelated (warm caches, allocator reuse,
CPU frequency state), so they are not independent samples. Confidence intervals
and tests in this module are meant to be applied to *per-process* summaries: one
value per independent process run. Per-iteration traces are only summarized
descriptively. See Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time"
(ISMM 2013).

Caveat: percentile-bootstrap intervals under-cover when there are only a handful
of samples. With ~5 process repeats treat the CI as indicative; use >=10 repeats
for claims that matter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import mannwhitneyu

Statistic = Callable[..., np.ndarray]


def _as_1d(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"expected a non-empty 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    min: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize(values: ArrayLike) -> Summary:
    """Descriptive statistics of a sample (sample std, ddof=1)."""
    arr = _as_1d(values)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    return Summary(
        n=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        min=float(arr.min()),
        p50=float(p50),
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        max=float(arr.max()),
    )


def bootstrap_ci(
    values: ArrayLike,
    statistic: Statistic = np.median,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap CI of ``statistic`` over independent samples.

    ``statistic`` must accept an ``axis`` keyword (e.g. ``np.median``, ``np.mean``).
    """
    arr = _as_1d(values)
    if arr.size < 2:
        raise ValueError("bootstrap needs at least 2 independent samples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = statistic(arr[idx], axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [tail, 1.0 - tail])
    return float(lo), float(hi)


def cliffs_delta(a: ArrayLike, b: ArrayLike) -> float:
    """P(a > b) - P(a < b) over all pairs; in [-1, 1], positive means ``a`` tends larger."""
    x, y = _as_1d(a), _as_1d(b)
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


@dataclass(frozen=True)
class Comparison:
    """``b`` relative to ``a``: ``ratio < 1`` means ``b`` has the lower median."""

    n_a: int
    n_b: int
    median_a: float
    median_b: float
    ratio: float
    ratio_ci: tuple[float, float]
    mannwhitney_u: float
    p_value: float
    cliffs_delta: float

    def to_dict(self) -> dict[str, float | int | tuple[float, float]]:
        return asdict(self)


def compare(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Compare two sets of independent per-process measurements (e.g. median latencies).

    Reports the ratio of medians with a bootstrap CI (each group resampled
    independently), a two-sided Mann-Whitney U test, and Cliff's delta.
    """
    x, y = _as_1d(a), _as_1d(b)
    if x.size < 2 or y.size < 2:
        raise ValueError("compare needs at least 2 independent samples per group")
    rng = np.random.default_rng(seed)
    bx = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    by = np.median(y[rng.integers(0, y.size, size=(n_boot, y.size))], axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(by / bx, [tail, 1.0 - tail])
    test = mannwhitneyu(x, y, alternative="two-sided")
    med_x, med_y = float(np.median(x)), float(np.median(y))
    return Comparison(
        n_a=int(x.size),
        n_b=int(y.size),
        median_a=med_x,
        median_b=med_y,
        ratio=med_y / med_x,
        ratio_ci=(float(lo), float(hi)),
        mannwhitney_u=float(test.statistic),
        p_value=float(test.pvalue),
        cliffs_delta=cliffs_delta(x, y),
    )
