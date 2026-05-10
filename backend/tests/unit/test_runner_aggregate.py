"""Unit tests for :mod:`ollama_evaluator.runner.aggregate` (Task 6.1).

Tests exercise each of the three public entry points:

* :func:`aggregate_metric_scores` — the numeric helper used by the
  :class:`MetricAggregate` builder. Covers the empty-list shortcut,
  the single-element contract (pstdev returns ``0.0``), and the
  general case against :func:`statistics.fmean` / :func:`pstdev`
  directly.

* :func:`build_model_aggregate` — the per-model roll-up. Covers
  status counters, the ``None``-skipping rule on performance fields
  (Requirement 6.5), the "empty results" edge case, and the
  error-isolation rule that excludes metrics with ``error != None``
  from the per-metric aggregate while still registering the metric
  name so the aggregate dict carries every observed metric
  (Requirement 7.5 — this is the unit-test counterpart to
  Property 14).

* :func:`build_all_aggregates` — the grouping front door. Covers
  model partitioning, lexicographic ordering, and the empty-input
  contract.
"""

from __future__ import annotations

import math
import statistics

from ollama_evaluator.models import (
    MetricResult,
    ModelAggregate,
    PerformanceMetrics,
    TestCaseResult,
)
from ollama_evaluator.runner.aggregate import (
    aggregate_metric_scores,
    build_all_aggregates,
    build_model_aggregate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perf(
    *,
    ttft_ms: float | None = 100.0,
    total_ms: float = 1000.0,
    prompt_tokens: int | None = 20,
    response_tokens: int | None = 50,
    tokens_per_second: float | None = 50.0,
) -> PerformanceMetrics:
    """Build a :class:`PerformanceMetrics` with sensible defaults."""
    return PerformanceMetrics(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        tokens_per_second=tokens_per_second,
    )


def _make_result(
    *,
    model: str = "m1",
    suite: str = "s1",
    test_case_id: str = "c1",
    repetition: int = 1,
    status: str = "pass",
    metrics: list[MetricResult] | None = None,
    performance: PerformanceMetrics | None = None,
) -> TestCaseResult:
    """Build a :class:`TestCaseResult` with sensible defaults."""
    return TestCaseResult(
        model=model,
        suite=suite,
        test_case_id=test_case_id,
        repetition=repetition,
        status=status,  # type: ignore[arg-type]
        response=None if status in ("error", "timeout") else "response",
        error_message="err" if status in ("error", "timeout") else None,
        performance=performance or _make_perf(),
        metrics=metrics or [],
    )


# ---------------------------------------------------------------------------
# aggregate_metric_scores
# ---------------------------------------------------------------------------


class TestAggregateMetricScores:
    def test_empty_list_returns_zeros(self) -> None:
        """Empty input returns the documented ``(0.0, 0.0, 0)`` triple."""
        assert aggregate_metric_scores([]) == (0.0, 0.0, 0)

    def test_single_element_stddev_is_zero(self) -> None:
        """``pstdev`` on a singleton is ``0.0`` — the aggregator delegates verbatim."""
        mean, stddev, count = aggregate_metric_scores([0.7])

        assert math.isclose(mean, 0.7)
        assert stddev == 0.0
        assert count == 1

    def test_multi_element_matches_statistics_module(self) -> None:
        """``mean`` and ``stddev`` match the :mod:`statistics` module directly."""
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]

        mean, stddev, count = aggregate_metric_scores(scores)

        assert math.isclose(mean, statistics.fmean(scores))
        assert math.isclose(stddev, statistics.pstdev(scores))
        assert count == 5


# ---------------------------------------------------------------------------
# build_model_aggregate
# ---------------------------------------------------------------------------


class TestBuildModelAggregate:
    def test_counts_all_four_statuses(self) -> None:
        """Status counters sum to ``len(results)`` across the four literals."""
        results = [
            _make_result(status="pass"),
            _make_result(status="pass"),
            _make_result(status="fail"),
            _make_result(status="error"),
            _make_result(status="timeout"),
        ]

        agg = build_model_aggregate("m1", results)

        assert agg.passed == 2
        assert agg.failed == 1
        assert agg.errored == 1
        assert agg.timed_out == 1
        # Sanity: counters sum to input length.
        assert (
            agg.passed + agg.failed + agg.errored + agg.timed_out == len(results)
        )

    def test_mean_ttft_skips_none_samples(self) -> None:
        """Only results with a non-``None`` ``ttft_ms`` contribute to the mean."""
        results = [
            _make_result(performance=_make_perf(ttft_ms=100.0)),
            _make_result(performance=_make_perf(ttft_ms=None)),
            _make_result(performance=_make_perf(ttft_ms=300.0)),
        ]

        agg = build_model_aggregate("m1", results)

        assert agg.mean_ttft_ms == statistics.fmean([100.0, 300.0])

    def test_mean_ttft_is_none_when_no_samples(self) -> None:
        """When every ``ttft_ms`` is ``None``, the mean is ``None`` (Requirement 6.5)."""
        results = [
            _make_result(performance=_make_perf(ttft_ms=None)),
            _make_result(performance=_make_perf(ttft_ms=None)),
        ]

        agg = build_model_aggregate("m1", results)

        assert agg.mean_ttft_ms is None

    def test_mean_tokens_per_second_skips_none_samples(self) -> None:
        results = [
            _make_result(performance=_make_perf(tokens_per_second=10.0)),
            _make_result(performance=_make_perf(tokens_per_second=None)),
            _make_result(performance=_make_perf(tokens_per_second=20.0)),
        ]

        agg = build_model_aggregate("m1", results)

        assert agg.mean_tokens_per_second == statistics.fmean([10.0, 20.0])

    def test_mean_total_ms_averages_over_all_results(self) -> None:
        """``total_ms`` is non-nullable and averaged over every result."""
        results = [
            _make_result(performance=_make_perf(total_ms=100.0)),
            _make_result(performance=_make_perf(total_ms=200.0)),
            _make_result(performance=_make_perf(total_ms=300.0)),
        ]

        agg = build_model_aggregate("m1", results)

        assert agg.mean_total_ms == 200.0

    def test_mean_total_ms_empty_results_is_zero(self) -> None:
        """Empty ``results`` yields ``mean_total_ms == 0.0`` (documented default)."""
        agg = build_model_aggregate("m1", [])

        assert agg.mean_total_ms == 0.0

    def test_per_metric_aggregate_averages_successful_scores(self) -> None:
        """Metrics without errors feed their scores into a :class:`MetricAggregate`."""
        results = [
            _make_result(
                metrics=[
                    MetricResult(name="exact-match", score=1.0, passed=True),
                    MetricResult(name="length-range", score=0.0, passed=False),
                ]
            ),
            _make_result(
                metrics=[
                    MetricResult(name="exact-match", score=0.0, passed=False),
                    MetricResult(name="length-range", score=1.0, passed=True),
                ]
            ),
        ]

        agg = build_model_aggregate("m1", results)

        assert set(agg.metric_aggregates) == {"exact-match", "length-range"}
        em = agg.metric_aggregates["exact-match"]
        assert em.count == 2
        assert em.mean == 0.5
        assert em.stddev == statistics.pstdev([1.0, 0.0])
        lr = agg.metric_aggregates["length-range"]
        assert lr.count == 2
        assert lr.mean == 0.5

    def test_error_metrics_excluded_from_aggregate(self) -> None:
        """Metrics with ``error != None`` are *not* averaged into the mean.

        This is the unit-test counterpart to Property 14 (metric error
        isolation): the aggregate still carries the metric name (so the
        dict is complete) but its ``count`` reflects only the non-error
        executions.
        """
        results = [
            _make_result(
                metrics=[
                    MetricResult(name="exact-match", score=1.0, passed=True),
                ]
            ),
            _make_result(
                metrics=[
                    MetricResult(
                        name="exact-match",
                        score=0.0,
                        passed=False,
                        error="boom",
                    ),
                ]
            ),
        ]

        agg = build_model_aggregate("m1", results)

        em = agg.metric_aggregates["exact-match"]
        assert em.count == 1
        assert em.mean == 1.0
        assert em.stddev == 0.0

    def test_metric_errored_on_every_execution_yields_zero_count(self) -> None:
        """A metric that errored on every execution still appears, with ``count == 0``."""
        results = [
            _make_result(
                metrics=[
                    MetricResult(
                        name="only-errored",
                        score=0.0,
                        passed=False,
                        error="boom",
                    ),
                ]
            ),
            _make_result(
                metrics=[
                    MetricResult(
                        name="only-errored",
                        score=0.0,
                        passed=False,
                        error="still boom",
                    ),
                ]
            ),
        ]

        agg = build_model_aggregate("m1", results)

        oe = agg.metric_aggregates["only-errored"]
        assert oe.count == 0
        assert oe.mean == 0.0
        assert oe.stddev == 0.0


# ---------------------------------------------------------------------------
# build_all_aggregates
# ---------------------------------------------------------------------------


class TestBuildAllAggregates:
    def test_empty_results_returns_empty_list(self) -> None:
        assert build_all_aggregates([]) == []

    def test_groups_by_model_in_sorted_order(self) -> None:
        """Models are partitioned and emitted in lexicographic order."""
        results = [
            _make_result(model="zebra", status="pass"),
            _make_result(model="apple", status="fail"),
            _make_result(model="mango", status="pass"),
            _make_result(model="apple", status="pass"),
        ]

        aggregates = build_all_aggregates(results)

        assert [a.model for a in aggregates] == ["apple", "mango", "zebra"]
        # The ``apple`` aggregate carries both of its results.
        apple = next(a for a in aggregates if a.model == "apple")
        assert apple.passed == 1
        assert apple.failed == 1

    def test_returns_model_aggregate_instances(self) -> None:
        """Each entry is a :class:`ModelAggregate` — not a dict or tuple."""
        aggregates = build_all_aggregates(
            [_make_result(model="m1"), _make_result(model="m2")]
        )
        assert all(isinstance(a, ModelAggregate) for a in aggregates)
