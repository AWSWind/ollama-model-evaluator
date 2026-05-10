"""Feature: ollama-model-evaluator, Property 19: Comparison over intersection.

For any pair of Run_Reports ``A`` and ``B``, let
``K_metric = {(model, metric)}`` present in both and ``K_perf = {model}``
present in both. Then:

* ``compare(A, B).metric_diffs`` is keyed exactly by ``K_metric`` with
  each entry satisfying ``diff == mean_b - mean_a``.
* ``compare(A, B).performance_diffs`` is keyed exactly by ``K_perf``.
* ``compare(A, B)`` raises :class:`NoCommonDimensionsError` if and
  only if ``K_metric == ∅`` *and* ``K_perf == ∅``.

Validates: Requirements 9.2, 9.3, 9.4.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from ollama_evaluator.compare import NoCommonDimensionsError, compare
from ollama_evaluator.models import RunReport

from .generators import run_reports


def _metric_keys(report: RunReport) -> set[tuple[str, str]]:
    return {
        (agg.model, metric)
        for agg in report.aggregates
        for metric in agg.metric_aggregates
    }


def _perf_keys(report: RunReport) -> set[str]:
    return {agg.model for agg in report.aggregates}


@given(a=run_reports(), b=run_reports())
@settings(max_examples=20, deadline=None)
def test_comparison_keys_and_diff_arithmetic(a: RunReport, b: RunReport) -> None:
    """**Validates: Requirements 9.2, 9.3, 9.4**

    Assert keys, diff arithmetic, and ``NoCommonDimensionsError``
    semantics on the reference intersections.
    """
    k_metric = _metric_keys(a) & _metric_keys(b)
    k_perf = _perf_keys(a) & _perf_keys(b)

    if not k_metric and not k_perf:
        with pytest.raises(NoCommonDimensionsError):
            compare(a, b)
        return

    report = compare(a, b)

    # Keyset equality.
    actual_metric_keys = {(d.model, d.metric) for d in report.metric_diffs}
    assert actual_metric_keys == k_metric, (
        f"metric_diffs keys {actual_metric_keys} != intersection {k_metric}"
    )
    actual_perf_keys = {d.model for d in report.performance_diffs}
    assert actual_perf_keys == k_perf, (
        f"performance_diffs keys {actual_perf_keys} != intersection {k_perf}"
    )

    # Diff arithmetic — ``mean_b - mean_a`` per ``(model, metric)``.
    a_models = {agg.model: agg for agg in a.aggregates}
    b_models = {agg.model: agg for agg in b.aggregates}
    for diff in report.metric_diffs:
        expected_a = a_models[diff.model].metric_aggregates[diff.metric].mean
        expected_b = b_models[diff.model].metric_aggregates[diff.metric].mean
        assert diff.mean_a == expected_a
        assert diff.mean_b == expected_b
        assert diff.diff == expected_b - expected_a

    # Performance arithmetic — ``total_ms_diff`` always concrete;
    # ``tps_diff`` nullable per the rule.
    for pdiff in report.performance_diffs:
        agg_a = a_models[pdiff.model]
        agg_b = b_models[pdiff.model]
        assert pdiff.mean_total_ms_a == agg_a.mean_total_ms
        assert pdiff.mean_total_ms_b == agg_b.mean_total_ms
        assert pdiff.total_ms_diff == agg_b.mean_total_ms - agg_a.mean_total_ms
        tps_a = agg_a.mean_tokens_per_second
        tps_b = agg_b.mean_tokens_per_second
        if tps_a is None or tps_b is None:
            assert pdiff.tps_diff is None
        else:
            assert pdiff.tps_diff == tps_b - tps_a
