"""Property-based invariants (Hypothesis) for the core numerical utilities.

These assert *laws* that must hold for all inputs, not just hand-picked examples:
dominance is a strict partial order, bootstrap CIs bracket their estimate,
Cliff's delta is bounded and antisymmetric, hypervolume ignores dominated points,
and deep_merge is a well-behaved config merge.
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from edge_ai_compression.benchmarking.statistics import bootstrap_ci, cliffs_delta
from edge_ai_compression.optimization.indicators import hypervolume
from edge_ai_compression.optimization.pareto.dominance import dominates
from edge_ai_compression.recipes.apply import deep_merge
from edge_ai_compression.theory.roofline import roofline, theoretical_min_latency_ms

_MAX = ("a",)
_MIN = ("b",)
finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
samples = st.lists(finite, min_size=2, max_size=100)


@given(a=finite, b=finite)
def test_dominance_irreflexive(a, b):
    p = {"a": a, "b": b}
    assert dominates(p, p, _MAX, _MIN) is False


@given(a1=finite, b1=finite, a2=finite, b2=finite)
def test_dominance_asymmetric(a1, b1, a2, b2):
    p = {"a": a1, "b": b1}
    q = {"a": a2, "b": b2}
    if dominates(p, q, _MAX, _MIN):
        assert not dominates(q, p, _MAX, _MIN)


@given(xs=samples, conf=st.floats(min_value=0.5, max_value=0.999))
@settings(max_examples=50)
def test_bootstrap_ci_brackets_point(xs, conf):
    ci = bootstrap_ci(xs, statistic="median", confidence=conf, n_resamples=300, seed=0)
    assert ci["low"] <= ci["point"] <= ci["high"]


@given(a=samples, b=samples)
@settings(max_examples=50)
def test_cliffs_delta_bounded_and_antisymmetric(a, b):
    d_ab = cliffs_delta(a, b)["delta"]
    d_ba = cliffs_delta(b, a)["delta"]
    assert -1.0 <= d_ab <= 1.0
    assert abs(d_ab + d_ba) < 1e-9


@given(
    px=st.floats(min_value=0.0, max_value=2.0),
    py=st.floats(min_value=0.0, max_value=2.0),
)
@settings(max_examples=50)
def test_hypervolume_nonnegative_and_dominated_insensitive(px, py):
    ref = [3.0, 3.0]
    front = [[px, py]]
    hv = hypervolume(front, ref)
    assert hv >= 0.0
    # A strictly-dominated extra point (worse in both, still within ref).
    worse = [min(px + 0.5, 3.0), min(py + 0.5, 3.0)]
    hv2 = hypervolume(front + [worse], ref)
    assert abs(hv - hv2) < 1e-9


@given(
    base=st.dictionaries(st.text(min_size=1, max_size=4), finite, max_size=5),
)
@settings(max_examples=50)
def test_deep_merge_identity_and_no_mutation(base):
    original = copy.deepcopy(base)
    merged = deep_merge(base, {})
    assert merged == base  # empty overlay is identity
    assert base == original  # base not mutated
    # Idempotence: merging the result with itself changes nothing.
    assert deep_merge(merged, merged) == merged


@given(
    intensity=positive,
    peak=positive,
    bw=positive,
)
@settings(max_examples=50)
def test_roofline_attainable_never_exceeds_peak(intensity, peak, bw):
    rl = roofline(intensity, peak_gflops=peak, mem_bandwidth_gbs=bw)
    assert rl["attainable_gflops"] <= peak + 1e-6


@given(flops=positive, byts=positive, peak=positive, bw=positive)
@settings(max_examples=50)
def test_latency_lower_bound_is_max_of_the_two(flops, byts, peak, bw):
    b = theoretical_min_latency_ms(flops, byts, peak_gflops=peak, mem_bandwidth_gbs=bw)
    assert b["lower_bound_ms"] == max(b["compute_bound_ms"], b["memory_bound_ms"])
