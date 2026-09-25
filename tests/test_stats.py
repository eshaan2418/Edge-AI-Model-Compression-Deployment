from __future__ import annotations

import numpy as np
import pytest

from edge_ai_compression.benchmarking.stats import (
    bootstrap_ci,
    cliffs_delta,
    compare,
    summarize,
)


def test_summarize_matches_numpy():
    x = np.random.default_rng(0).lognormal(size=1001)
    s = summarize(x)
    assert s.n == 1001
    assert s.mean == pytest.approx(x.mean())
    assert s.std == pytest.approx(x.std(ddof=1))
    assert s.p50 == pytest.approx(np.median(x))
    assert s.p95 == pytest.approx(np.percentile(x, 95))
    assert s.p99 == pytest.approx(np.percentile(x, 99))
    assert (s.min, s.max) == (x.min(), x.max())


def test_summarize_single_value_and_invalid_input():
    assert summarize([3.0]).std == 0.0
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize([1.0, float("nan")])
    with pytest.raises(ValueError):
        summarize([[1.0, 2.0]])


def test_bootstrap_ci_requires_two_samples():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])


def test_bootstrap_ci_coverage_near_nominal():
    # Lognormal(0, 0.5) has median exactly 1.0.
    rng = np.random.default_rng(1)
    trials, hits = 200, 0
    for i in range(trials):
        sample = rng.lognormal(0.0, 0.5, size=30)
        lo, hi = bootstrap_ci(sample, n_boot=2000, seed=i)
        assert lo <= np.median(sample) <= hi
        hits += lo <= 1.0 <= hi
    # Percentile bootstrap slightly under-covers at n=30; nominal is 0.95.
    assert 0.88 <= hits / trials <= 0.99


def test_cliffs_delta_known_values():
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_compare_detects_known_shift():
    rng = np.random.default_rng(2)
    a = rng.lognormal(np.log(10.0), 0.05, size=10)
    b = rng.lognormal(np.log(8.0), 0.05, size=10)
    c = compare(a, b)
    assert c.p_value < 0.01
    assert c.ratio == pytest.approx(0.8, rel=0.05)
    assert c.ratio_ci[0] <= 0.8 <= c.ratio_ci[1]
    assert c.cliffs_delta > 0.9


def test_compare_false_positive_rate_on_identical_distributions():
    rng = np.random.default_rng(3)
    trials = 200
    rejections = sum(
        compare(rng.normal(10, 1, size=10), rng.normal(10, 1, size=10), n_boot=200).p_value < 0.05
        for _ in range(trials)
    )
    assert rejections / trials <= 0.1


def test_compare_requires_two_samples_per_group():
    with pytest.raises(ValueError):
        compare([1.0], [1.0, 2.0])
