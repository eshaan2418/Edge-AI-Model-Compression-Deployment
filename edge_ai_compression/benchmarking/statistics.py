"""Statistically rigorous comparison of latency samples.

Point estimates lie: a "12ms vs 11ms" headline means nothing without knowing the
spread and whether the gap survives noise. This module treats a latency
measurement as a *sample* and answers the questions a careful reviewer asks:

* How uncertain is the estimate? — bootstrap confidence intervals.
* Is model A actually faster than model B, or is it noise? — a distribution-free
  significance test (Mann-Whitney U) plus an effect size (Cliff's delta).
* How big is the speedup, with error bars? — a bootstrap CI on the ratio of
  medians.

Everything is nonparametric (latency is skewed and heavy-tailed, so we do not
assume normality) and fully deterministic given a seed.

Note: these are wall-clock measurements on the benchmarking host, not the target
device — see the hardware profiles for device budgets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

# Cliff's delta magnitude thresholds (Romano et al., 2006).
_NEGLIGIBLE = 0.147
_SMALL = 0.33
_MEDIUM = 0.474


def _as_array(samples: object) -> np.ndarray:
    arr = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("samples must be a non-empty 1-D sequence of numbers")
    return arr


def _statistic_fn(name: str):
    if name == "median":
        return np.median
    if name == "mean":
        return np.mean
    raise ValueError(f"Unknown statistic '{name}' (use 'median' or 'mean').")


def bootstrap_ci(
    samples: object,
    *,
    statistic: str = "median",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a statistic of ``samples``.

    Resamples with replacement ``n_resamples`` times and takes the empirical
    percentiles of the bootstrap distribution. Deterministic given ``seed``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arr = _as_array(samples)
    fn = _statistic_fn(statistic)
    point = float(fn(arr))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot = fn(arr[idx], axis=1)
    alpha = 1.0 - confidence
    low = float(np.percentile(boot, 100.0 * alpha / 2.0))
    high = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {"point": point, "low": low, "high": high, "confidence": confidence}


def coefficient_of_variation(samples: object) -> float:
    """Std / mean — a unitless measure of relative dispersion (noise)."""
    arr = _as_array(samples)
    mean = float(np.mean(arr))
    if mean == 0.0:
        return float("inf")
    return float(np.std(arr) / mean)


def outlier_fraction(samples: object, *, k: float = 3.0) -> float:
    """Fraction of samples more than ``k`` robust std-devs from the median.

    Uses the MAD (scaled to a normal-consistent sigma) as the scale. When more
    than half the samples are identical the MAD degenerates to 0; in that case we
    fall back to the ordinary standard deviation so genuine outliers are still
    caught rather than silently masked.
    """
    arr = _as_array(samples)
    med = np.median(arr)
    dev = np.abs(arr - med)
    # 1.4826 scales MAD to a normal-consistent standard-deviation estimate.
    scale = 1.4826 * np.median(dev)
    if scale == 0.0:
        scale = float(np.std(arr))
    if scale == 0.0:
        return 0.0
    return float(np.mean(dev > k * scale))


def mann_whitney_u(a: object, b: object) -> dict[str, float]:
    """Two-sided Mann-Whitney U test — is one distribution stochastically larger?

    Distribution-free (no normality assumption), which suits skewed latency data.
    """
    aa, bb = _as_array(a), _as_array(b)
    result = stats.mannwhitneyu(aa, bb, alternative="two-sided")
    return {"u": float(result.statistic), "p_value": float(result.pvalue)}


def cliffs_delta(a: object, b: object) -> dict[str, Any]:
    """Cliff's delta effect size in [-1, 1] with a qualitative magnitude label.

    delta = P(a > b) - P(a < b). Positive means ``a`` tends to be larger.
    Computed via rank statistics in O(n log n) rather than the O(n*m) pairwise sum.
    """
    aa, bb = _as_array(a), _as_array(b)
    n, m = aa.size, bb.size
    # Rank all values together; the summed ranks of `a` give the U statistic,
    # from which the "greater-than" probability follows.
    combined = np.concatenate([aa, bb])
    ranks = stats.rankdata(combined)
    rank_sum_a = float(np.sum(ranks[:n]))
    u_a = rank_sum_a - n * (n + 1) / 2.0  # # of (a_i > b_j) pairs, ties as 0.5
    delta = (2.0 * u_a) / (n * m) - 1.0
    delta = float(np.clip(delta, -1.0, 1.0))

    mag = abs(delta)
    if mag < _NEGLIGIBLE:
        label = "negligible"
    elif mag < _SMALL:
        label = "small"
    elif mag < _MEDIUM:
        label = "medium"
    else:
        label = "large"
    return {"delta": delta, "magnitude": label}


def _bootstrap_ratio_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap CI for median(a) / median(b) (the speedup factor of b over a)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_resamples, a.size))
    ib = rng.integers(0, b.size, size=(n_resamples, b.size))
    med_a = np.median(a[ia], axis=1)
    med_b = np.median(b[ib], axis=1)
    ratio = med_a / np.clip(med_b, 1e-12, None)
    alpha = 1.0 - confidence
    point = float(np.median(a) / max(float(np.median(b)), 1e-12))
    return {
        "point": point,
        "low": float(np.percentile(ratio, 100.0 * alpha / 2.0)),
        "high": float(np.percentile(ratio, 100.0 * (1.0 - alpha / 2.0))),
    }


def compare_latency(
    a: object,
    b: object,
    *,
    label_a: str = "A",
    label_b: str = "B",
    confidence: float = 0.95,
    alpha: float = 0.05,
    seed: int = 0,
    n_resamples: int = 2000,
) -> dict[str, Any]:
    """Full A/B latency comparison with CIs, a significance test, and a verdict.

    ``a`` and ``b`` are raw per-run latency samples (milliseconds). Returns a
    JSON-serializable report; lower latency is better. The speedup factor is
    ``median(a) / median(b)`` — i.e. how many times faster ``b`` is than ``a``.
    """
    aa, bb = _as_array(a), _as_array(b)

    stats_a = {
        "n": int(aa.size),
        "median_ms": bootstrap_ci(
            aa, statistic="median", confidence=confidence, n_resamples=n_resamples, seed=seed
        ),
        "cv": coefficient_of_variation(aa),
        "outlier_fraction": outlier_fraction(aa),
    }
    stats_b = {
        "n": int(bb.size),
        "median_ms": bootstrap_ci(
            bb, statistic="median", confidence=confidence, n_resamples=n_resamples, seed=seed + 1
        ),
        "cv": coefficient_of_variation(bb),
        "outlier_fraction": outlier_fraction(bb),
    }

    mw = mann_whitney_u(aa, bb)
    delta = cliffs_delta(aa, bb)
    speedup = _bootstrap_ratio_ci(
        aa, bb, confidence=confidence, n_resamples=n_resamples, seed=seed + 2
    )

    significant = mw["p_value"] < alpha
    faster = label_b if speedup["point"] > 1.0 else label_a
    slower = label_a if faster == label_b else label_b
    factor = speedup["point"] if speedup["point"] >= 1.0 else 1.0 / max(speedup["point"], 1e-12)
    if significant and delta["magnitude"] != "negligible":
        verdict = (
            f"{faster} is significantly faster than {slower} "
            f"({factor:.2f}x median, p={mw['p_value']:.4g}, "
            f"Cliff's delta={delta['delta']:+.3f} [{delta['magnitude']}])."
        )
    else:
        verdict = (
            f"No significant latency difference between {label_a} and {label_b} "
            f"(p={mw['p_value']:.4g}, Cliff's delta={delta['delta']:+.3f} [{delta['magnitude']}])."
        )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "confidence": confidence,
        "alpha": alpha,
        label_a: stats_a,
        label_b: stats_b,
        "speedup_b_over_a": speedup,
        "mann_whitney": mw,
        "cliffs_delta": delta,
        "significant": significant,
        "verdict": verdict,
    }
